import os

# Smart environment setup
import sys
from pathlib import Path
config_dir = Path(__file__).parent.parent / "config"
if config_dir.exists():
    sys.path.insert(0, str(config_dir))
    from env_config import setup_environment
    setup_environment()
else:
    # Fallback if config not found - auto-detect project root
    from pathlib import Path
    current_path = Path(__file__).resolve()
    project_root = None
    
    # Walk up directory tree to find project root
    for parent in current_path.parents:
        if parent.name == 'Adaptive-RAG':
            project_root = str(parent.parent)
            break
        if any((parent / marker).exists() for marker in [
            'adaptive_rag_benchmark', 'classifier', 'scaled_silver_labeling'
        ]):
            project_root = str(parent.parent) if parent.name == 'Adaptive-RAG' else str(parent)
            break
    
    if not project_root:
        project_root = os.environ.get('PROJECT_ROOT')
        if not project_root:
            raise RuntimeError("Could not auto-detect PROJECT_ROOT. Please set PROJECT_ROOT environment variable.")
    
    os.environ['PROJECT_ROOT'] = project_root
    cache_dir = f"{project_root}/Adaptive-RAG/.cache"
    os.environ['HF_HOME'] = f"{cache_dir}/huggingface"
    os.environ['HF_DATASETS_CACHE'] = f"{cache_dir}/huggingface"
    os.environ['TRANSFORMERS_CACHE'] = f"{cache_dir}/transformers"
import time
import warnings
import json
import re
import asyncio
from functools import lru_cache

# Basic warning suppression
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5Config,
)


@lru_cache(maxsize=None)
def get_model_and_tokenizer():

    model_shortname = os.environ["MODEL_NAME"]

    valid_model_shortnames = [
        "flan-t5-large",
    ]
    assert model_shortname in valid_model_shortnames, f"Model name {model_shortname} not in {valid_model_shortnames}"

    if model_shortname.startswith("flan-t5") and "bf16" not in model_shortname:
        model_name = "google/" + model_shortname
        
        # Load config and modify max_position_embeddings
        config = T5Config.from_pretrained(model_name)
        config.max_position_embeddings = 1024  # Increase from default 512 to 1024
        
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main", device_map=hf_device_map, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Update tokenizer to support longer sequences
        tokenizer.model_max_length = 1024

    elif model_shortname.startswith("flan-t5") and model_shortname.endswith("-bf16"):

        assert torch.cuda.is_bf16_supported()
        assert is_torch_bf16_gpu_available()
        model_name = "google/" + model_shortname.replace("-bf16", "")
        
        # Load config and modify max_position_embeddings
        config = T5Config.from_pretrained(model_name)
        config.max_position_embeddings = 1024  # Increase from default 512 to 1024
        
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map=hf_device_map, torch_dtype=torch.bfloat16, config=config
        )
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        # Update tokenizer to support longer sequences
        tokenizer.model_max_length = 1024



    return model, tokenizer


class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert (
            len(self.tokenizer.encode(eos_text)) < 10
        ), "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        condition1 = decoded_text.endswith(self.eos_text)
        condition2 = decoded_text.strip().endswith(self.eos_text.strip())
        return condition1 or condition2


def parse_conversation_prompt(prompt: str):
    """
    Parse a prompt to extract conversation messages for chat template usage.
    
    Returns:
        (messages, is_conversation): If it's a conversation format, returns the messages list.
        Otherwise returns (None, False) to use raw prompt.
    """
    try:
        # Try to detect if this is a structured conversation
        # Look for patterns like "system\n" and "user\n" and "assistant\n"
        lines = prompt.strip().split('\n')
        
        # Check if it looks like a conversation format
        has_system = any(line.strip().lower() == 'system' for line in lines)
        has_user = any(line.strip().lower() == 'user' for line in lines)
        
        if has_system or has_user:
            # Parse as conversation
            messages = []
            current_role = None
            current_content = []
            
            for line in lines:
                line_lower = line.strip().lower()
                if line_lower in ['system', 'user', 'assistant']:
                    # Save previous message
                    if current_role and current_content:
                        content = '\n'.join(current_content).strip()
                        if content:
                            messages.append({"role": current_role, "content": content})
                    
                    # Start new message
                    current_role = line_lower
                    current_content = []
                else:
                    # Add to current message content
                    if line.strip():  # Skip empty lines
                        current_content.append(line)
            
            # Add final message
            if current_role and current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    messages.append({"role": current_role, "content": content})
            
            if messages:
                return messages, True
        
        # Fallback: treat as a simple user message
        return [{"role": "user", "content": prompt.strip()}], True
        
    except Exception:
        # If parsing fails, use raw prompt
        return None, False





app = FastAPI()

# 🔒 CONCURRENCY CONTROL: Prevent server overload from simultaneous requests
# Allow only 1 concurrent generation per server to prevent memory contention and hanging
generation_semaphore = asyncio.Semaphore(1)


@app.get("/")
async def index():
    model_shortname = os.environ["MODEL_NAME"]
    return {"message": f"Hello! This is a server for {model_shortname}. " "Go to /generate/ for generation requests."}


from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str
    max_input: Optional[int] = None
    max_length: int = 200
    max_tokens: Optional[int] = None  # Accept both max_length and max_tokens
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 1.0
    num_return_sequences: int = 1
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    eos_text: Optional[str] = None
    keep_prompt: bool = False

@app.post("/generate/")
async def generate(request: GenerateRequest):
    # 🔒 ACQUIRE SEMAPHORE: Serialize generation to prevent hanging
    async with generation_semaphore:
        start_time = time.time()

        model_shortname = os.environ["MODEL_NAME"]

        model, tokenizer = get_model_and_tokenizer()
        
        # Handle max_tokens vs max_length compatibility
        max_length = request.max_tokens if request.max_tokens is not None else request.max_length
        
        # Use max_input if provided, otherwise use tokenizer's updated model_max_length
        actual_max_input = request.max_input if request.max_input is not None else tokenizer.model_max_length
        
        # Handle input encoding consistently with better device management
        try:
            # Use model.device instead of assuming cuda
            inputs = tokenizer.encode(request.prompt, return_tensors="pt", max_length=actual_max_input, truncation=True).to(model.device)
        except Exception as e:
            raise Exception(f"Failed to encode input and move to model device: {e}")

        stopping_criteria_list = StoppingCriteriaList()
        if request.eos_text:
            stopping_criteria = EOSReachedCriteria(tokenizer=tokenizer, eos_text=request.eos_text)
            stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

        # T0pp, ul2 and flan are the only encoder-decoder model, and so don't have prompt part in its generation.
        is_encoder_decoder = model_shortname in ["T0pp", "ul2"] or model_shortname.startswith("flan-t5")
        
        temperature = request.temperature
        do_sample = request.do_sample
        
        # 🚀 GENERATION: Now protected by semaphore to prevent concurrent hanging
        generated_output = model.generate(
            inputs,
            max_new_tokens=max_length,
            min_length=request.min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences,
            return_dict_in_generate=True,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            stopping_criteria=stopping_criteria_list,
            output_scores=False,  # make it configurable later. It turns in generated_output["scores"]
        )
        generated_ids = generated_output["sequences"]
        
        # Handle response extraction consistently
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_num_tokens = [len(generated_ids_) for generated_ids_ in generated_ids]
        
        # Original logic for prompt removal (FLAN-T5 and fallback)
        if not request.keep_prompt and not is_encoder_decoder:
            generated_texts = [
                generated_text[generated_text.index(request.prompt) + len(request.prompt) :] if request.prompt in generated_text else generated_text 
                for generated_text in generated_texts
            ]
        elif request.keep_prompt and is_encoder_decoder:
            generated_texts = [request.prompt + generated_text for generated_text in generated_texts]

        end_time = time.time()
        run_time_in_seconds = end_time - start_time
        return {
            "generated_num_tokens": generated_num_tokens,
            "generated_texts": generated_texts,
            "run_time_in_seconds": run_time_in_seconds,
            "model_name": model_shortname,
        }


print("\nLoading model and tokenizer.")
get_model_and_tokenizer()
print("Loaded model and tokenizer.\n")
