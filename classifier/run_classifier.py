#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os

# Set up environment variables
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
# Auto-detect project root if not set
if 'PROJECT_ROOT' not in os.environ:
    from pathlib import Path
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == 'Adaptive-RAG':
            os.environ['PROJECT_ROOT'] = str(parent.parent)
            break
        if any((parent / marker).exists() for marker in [
            'adaptive_rag_benchmark', 'classifier', 'scaled_silver_labeling'
        ]):
            os.environ['PROJECT_ROOT'] = str(parent.parent) if parent.name == 'Adaptive-RAG' else str(parent)
            break
    if 'PROJECT_ROOT' not in os.environ:
        raise RuntimeError("Could not auto-detect PROJECT_ROOT. Please set PROJECT_ROOT environment variable.")

os.environ['HF_HOME'] = f"{os.environ['PROJECT_ROOT']}/Adaptive-RAG/.cache/huggingface"
import random
from pathlib import Path
from typing import List, Optional, Tuple
import copy
#from utils_qa import *
import pickle

import datasets
import nltk
import numpy as np
import torch

# Monkey patch torch.compile to work around Python 3.12+ compatibility issues with ModernBERT
def no_op_compile(func=None, **kwargs):
    """No-op replacement for torch.compile that just returns the function unchanged."""
    if func is None:
        return lambda f: f
    return func

# Replace torch.compile with no-op version
torch.compile = no_op_compile

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint


from torch.nn import CrossEntropyLoss

from utils import *


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

##
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

option_to_label = {
    'A': 0,
    'B': 1,
    'C': 2,
}

label_to_option = {
    0: 'A',
    1: 'B',
    2: 'C',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a QA task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after "
            "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--do_eval", action="store_true", help="To do eval on the question answering model")
    parser.add_argument("--do_train", action="store_true", help="To do train on the question answering model")
    # data col
    parser.add_argument(
        "--train_column",
        type=str,
        default='train',
        help="The name of the train column in the datasets.",
    )
    parser.add_argument(
        "--val_column",
        type=str,
        default='validation',
        help="The name of the validation column in the datasets.",
    )
    parser.add_argument(
        "--test_column",
        type=str,
        default='test',
        help="The name of the test column in the datasets.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--val_max_answer_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=f"{os.environ['PROJECT_ROOT']}/Adaptive-RAG/.cache",
        help="Directory to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading pretrained models",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default='question',
        help="The name of the column in the datasets containing the questions (for question answering).",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default='answers',
        help="The name of the column in the datasets containing the answers (for question answering).",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training.")
    parser.add_argument("--optim", type=str, default="adamw", help="Optimizer to use (adamw, adamw_torch_fused, adamw_8bit, etc.).")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    args = parser.parse_args()

    return args


def run_validation(model, eval_dataloader, eval_examples, eval_dataset, args, accelerator, tokenizer, is_bert_model, epoch=None):
    """Run validation and return metrics"""
    logger.info("***** Running Validation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    if args.val_max_answer_length is None:
        args.val_max_answer_length = args.max_answer_length

    gen_kwargs = {
        "max_length": args.val_max_answer_length,
        #'no_repeat_ngram_size':2
        #"num_beams": args.num_beams,
    }

    # inference
    model.eval()
    predictions = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            if is_bert_model:
                # BERT-style classification
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits
                preds_labels = torch.argmax(logits, dim=-1).cpu().numpy()
                preds = [label_to_option[pred] for pred in preds_labels]
                
                labels = batch["labels"]
                labels = accelerator.gather_for_metrics(labels)
                labels = labels.cpu().numpy()
            else:
                # T5-style generation
                scores = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs,
                ).scores[0]

                probs = (
                    torch.nn.functional.softmax(
                        torch.stack([
                            scores[:, tokenizer('A').input_ids[0]],
                            scores[:, tokenizer('B').input_ids[0]],
                            scores[:, tokenizer('C').input_ids[0]],
                        ]), dim=0,
                    ).detach().cpu().numpy()
                )

                preds_labels = np.argmax(probs, 0)
                preds = [label_to_option[pred] for pred in preds_labels]

                labels = batch["labels"]
                labels = accelerator.gather_for_metrics(labels)
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            predictions = predictions + preds

    gold_answers = eval_examples['answer']
    
    # Create validation directory if it doesn't exist
    validation_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    # Save detailed results
    dict_id_pred_results = {qid : {'prediction': pred, 'answer' : ans, 'dataset_name' : data} for qid, pred, ans, data in zip(eval_examples['id'], predictions, gold_answers, eval_examples['dataset_name'])}
    
    # Save with epoch suffix if provided
    if epoch is not None:
        results_file = os.path.join(validation_dir, f"dict_id_pred_results_epoch_{epoch}.json")
    else:
        results_file = os.path.join(validation_dir, "dict_id_pred_results.json")
    
    with open(results_file, "w") as f:
        json.dump(dict_id_pred_results, f, indent=4)

    assert len(gold_answers) == len(predictions)

    # Calculate metrics
    final_acc_score = calculate_accuracy(gold_answers, predictions)
    final_eval_results = {'final_acc_score' : final_acc_score}
    
    # Acc per class
    final_eval_results_perClass = calculate_accuracy_perClass(gold_answers, predictions)
    
    # Combine results
    combined_results = {
        'epoch': epoch,
        'accuracy': final_acc_score,
        'per_class_metrics': final_eval_results_perClass
    }

    logger.info(f"Validation Results - Epoch {epoch if epoch is not None else 'Final'}: Accuracy = {final_acc_score:.4f}")
    logger.info(f"Per-class metrics: {final_eval_results_perClass}")
    
    # Save epoch-specific results
    if epoch is not None:
        epoch_results_file = os.path.join(validation_dir, f"eval_results_epoch_{epoch}.json")
        with open(epoch_results_file, "w") as f:
            json.dump(combined_results, f, indent=4)
    
    # Also save as final results (will be overwritten each epoch, but ensures we have final results)
    with open(os.path.join(validation_dir, "final_eval_results.json"), "w") as f:
        json.dump(final_eval_results, f)

    with open(os.path.join(validation_dir, "final_eval_results_perClass.json"), "w") as f:
        json.dump(final_eval_results_perClass, f, indent=4)
    
    return combined_results


def main():
    args = parse_args()
    

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    # Add bf16 support if requested
    mixed_precision = "bf16" if args.bf16 else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        mixed_precision=mixed_precision,
        **accelerator_log_kwargs
    )
    
    device = accelerator.device

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )


    # Make one log on every process with the configuration for debugging.
    # TODO
    # Setup logging
    logging.basicConfig(        
        filename=args.output_dir+'/logs.log', # 
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )

    #logger.info(accelerator.state, main_process_only=False)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logger.info(args)


    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.do_eval:
            extension = args.validation_file.split(".")[-1]
        else:
            extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # load model and tokenizer
    model, tokenizer, is_bert_model = load_model(args)

        
    if args.do_train:
        if args.train_column not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets[args.train_column]

        if args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(args.max_train_samples))


        # Create train feature from dataset
        with accelerator.main_process_first():
            preprocess_func = preprocess_features_function_bert if is_bert_model else preprocess_features_function
            train_dataset = train_dataset.map(
                preprocess_func, 
                fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            if args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation, We select only specified max samples
                train_dataset = train_dataset.select(range(args.max_train_samples))
    
    
    if args.do_eval or args.do_train:  # Load validation data for training monitoring
        if args.val_column not in raw_datasets:
            if args.do_eval:
                raise ValueError("--do_eval requires a validation dataset")
            else:
                logger.warning("No validation dataset found, skipping validation during training")
                args.do_eval = False
        else:
            eval_examples = raw_datasets[args.val_column]

            if args.max_eval_samples is not None:
                # We will select sample from whole data
                eval_examples = eval_examples.select(range(args.max_eval_samples))
            # Validation Feature Creation
            with accelerator.main_process_first():
                preprocess_func = preprocess_features_function_bert if is_bert_model else preprocess_features_function
                eval_dataset = eval_examples.map(
                    preprocess_func, 
                    fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=eval_examples.column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

            if args.max_eval_samples is not None:
                # During Feature creation dataset samples might increase, we will select required samples again
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if is_bert_model:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )

    if args.do_train:
        train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])
        train_dataloader = DataLoader(
            train_dataset_for_model, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )

    if args.do_eval:
        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])        
        eval_dataloader = DataLoader(
            eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # Choose optimizer based on args.optim
    if args.optim == "adamw_torch_fused":
        try:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, 
                lr=args.learning_rate, 
                fused=True,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            logger.info("Using fused AdamW optimizer")
        except RuntimeError as e:
            if "fused=True" in str(e):
                logger.warning(f"Fused optimizer failed ({e}), falling back to regular AdamW")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters, 
                    lr=args.learning_rate,
                    eps=1e-8,
                    betas=(0.9, 0.999)
                )
            else:
                raise e
    elif args.optim == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate)
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to regular AdamW")
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999)
        )


    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    if args.do_train:
        train_dataloader = accelerator.prepare(
            train_dataloader
        )

    if args.do_eval:
        eval_dataloader = accelerator.prepare(
            eval_dataloader
        )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("no_trainer", experiment_config)

    # Train!
    if args.do_train:
        
        args.max_train_steps, args.num_train_epochs, lr_scheduler_train = prepare_scheduler(args, accelerator, train_dataloader, optimizer, args.max_train_steps, args.num_train_epochs)

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Store training history for analysis
        training_history = []

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0
            train_predictions = []
            train_labels = []
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler_train.step()
                    optimizer.zero_grad()

                    # logger.info("Loss:{} ".format(loss))

                    # We keep track of the loss at each epoch
                    total_loss = total_loss + loss.cpu().detach().float()
                    
                    # Collect predictions and labels for training accuracy
                    if is_bert_model:
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        preds = [label_to_option[pred] for pred in preds]
                        labels = batch["labels"].cpu().numpy()
                        labels = [label_to_option[label] for label in labels]
                        train_predictions.extend(preds)
                        train_labels.extend(labels)
                    else:
                        # For T5-style models, get predictions from logits
                        # We'll use the same approach as in validation but simpler
                        try:
                            # Get logits from the model outputs
                            if hasattr(outputs, 'logits') and outputs.logits is not None:
                                # For T5, we need to look at the first token predictions
                                logits = outputs.logits[:, 0, :]  # First token position
                                
                                # Get probabilities for A, B, C tokens
                                scores = torch.stack([
                                    logits[:, tokenizer('A').input_ids[0]],
                                    logits[:, tokenizer('B').input_ids[0]],
                                    logits[:, tokenizer('C').input_ids[0]],
                                ], dim=1)
                                
                                preds_labels = torch.argmax(scores, dim=1).cpu().numpy()
                                preds = [label_to_option[pred] for pred in preds_labels]
                                
                                # Get true labels
                                labels = batch["labels"].cpu().numpy()
                                if args.ignore_pad_token_for_loss:
                                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                                # Decode labels to get actual answers
                                label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                                labels = [text.strip() if text.strip() in ['A', 'B', 'C'] else 'A' for text in label_texts]
                                
                                train_predictions.extend(preds)
                                train_labels.extend(labels)
                        except Exception as e:
                            # If there's any issue with T5 training accuracy, skip it
                            pass

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                # Only create step checkpoints if explicitly requested (prevent step_0 creation)
                if isinstance(checkpointing_steps, int) and checkpointing_steps > 0:
                    if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            avg_epoch_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} Loss: {avg_epoch_loss}")
            
            # Calculate training accuracy
            if train_predictions and train_labels and len(train_predictions) == len(train_labels):
                train_accuracy = calculate_accuracy(train_labels, train_predictions)
            else:
                train_accuracy = None

            # Run validation after each epoch
            validation_results = None
            if args.do_eval:
                logger.info(f"Running validation after epoch {epoch}")
                validation_results = run_validation(
                    model, eval_dataloader, eval_examples, eval_dataset, 
                    args, accelerator, tokenizer, is_bert_model, epoch=epoch
                )
                
                # Print to terminal for immediate visibility
                if train_accuracy is not None:
                    print(f"Epoch {epoch} - Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {validation_results['accuracy']:.4f}")
                else:
                    print(f"Epoch {epoch} - Training Accuracy: N/A, Validation Accuracy: {validation_results['accuracy']:.4f}")
                
                # Log epoch summary
                logger.info(f"=== EPOCH {epoch} SUMMARY ===")
                logger.info(f"Training Loss: {avg_epoch_loss:.6f}")
                logger.info(f"Validation Accuracy: {validation_results['accuracy']:.4f}")
                logger.info(f"Per-class Accuracy: A={validation_results['per_class_metrics']['A (zero) acc']:.2f}%, B={validation_results['per_class_metrics']['B (single) acc']:.2f}%, C={validation_results['per_class_metrics']['C (multi) acc']:.2f}%")
                
                # Store training history
                training_history.append({
                    'epoch': epoch,
                    'train_loss': float(avg_epoch_loss),
                    'val_accuracy': validation_results['accuracy'],
                    'val_per_class': validation_results['per_class_metrics']
                })
                
                # Save training history
                history_file = os.path.join(args.output_dir, "training_history.json")
                with open(history_file, "w") as f:
                    json.dump(training_history, f, indent=4)

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Suppress verbose saving messages during training
                old_level = transformers.utils.logging.get_verbosity()
                transformers.utils.logging.set_verbosity_error()
                
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )
                
                # Restore logging level
                transformers.utils.logging.set_verbosity(old_level)


            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Suppress verbose saving messages during training (not final save)
                if epoch < args.num_train_epochs - 1:
                    # Temporarily suppress transformers logging
                    old_level = transformers.utils.logging.get_verbosity()
                    transformers.utils.logging.set_verbosity_error()
                    
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
                
                # Restore logging level if it was suppressed
                if epoch < args.num_train_epochs - 1:
                    transformers.utils.logging.set_verbosity(old_level)

        # Final training summary
        if args.do_eval and training_history:
            logger.info("=== TRAINING COMPLETED ===")
            logger.info("Training History Summary:")
            for i, hist in enumerate(training_history):
                logger.info(f"Epoch {i}: Loss={hist['train_loss']:.6f}, Val_Acc={hist['val_accuracy']:.4f}")
            
            best_epoch = max(training_history, key=lambda x: x['val_accuracy'])
            logger.info(f"Best validation accuracy: {best_epoch['val_accuracy']:.4f} at epoch {best_epoch['epoch']}")

    # Final Validation (if not done during training)
    if args.do_eval and not args.do_train:
        final_results = run_validation(
            model, eval_dataloader, eval_examples, eval_dataset, 
            args, accelerator, tokenizer, is_bert_model, epoch=None
        )
        print(f"Final Evaluation Results: {final_results}")


if __name__ == "__main__":
    main()
