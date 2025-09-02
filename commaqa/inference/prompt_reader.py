"""
Prompt Engineering Utilities for CommaQA Inference

This module provides robust functions for dynamically reading, constructing, and
truncating prompts for Large Language Model (LLM) inference. Its primary
purpose is to manage few-shot examples and ensure that the final prompt,
including the test query and space for generation, fits within the LLM's
context window.

Key Features:
- Reading prompts from single or multiple files.
- Filtering few-shot examples based on metadata (e.g., skill, difficulty).
- Dynamically removing examples to meet model length constraints.
- Multiple strategies for example removal (e.g., removing the last, longest).
- Caching tokenizers for efficiency.

Relation to other modules:
- This is a core utility for the `commaqa.inference` pipeline.
- Other modules (e.g., an inference runner) will call `read_prompt` to
  load a base prompt with few-shot examples.
- The output of this module is typically combined with a final test question
  before being passed to an LLM for generation.
- `fit_prompt_into_given_limit` is a more generic function that can be used
  by any part of the system that constructs prompts programmatically and
  needs to enforce a length limit.
"""
import os
import json
import copy
import random
from typing import Dict, List, Union
from functools import lru_cache

random.seed(13370)  # Don't change this for reproducibility.


@lru_cache(maxsize=15)
def get_tokenizer(model_name):
    """
    Loads and caches a Hugging Face tokenizer.

    Using @lru_cache prevents reloading the same tokenizer multiple times,
    which can be slow.

    Args:
        model_name (str): The name of the tokenizer model (e.g., "Qwen/Qwen2.5-7B-Instruct").

    Returns:
        transformers.PreTrainedTokenizer: The loaded tokenizer instance.
    """
    from transformers import AutoTokenizer

    # Avoids a warning from the transformers library.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return AutoTokenizer.from_pretrained(model_name)


def read_prompt(
    file_path: Union[str, List[str]] = "",
    filter_by_key_values: Dict[str, List[str]] = None,
    metadata_prefix: str = "# METADATA: ",
    order_by_key: str = None,
    test_to_train_length_scale: int = 1,
    estimated_generation_length: int = 500,
    shuffle: bool = False,
    model_length_limit: int = 8000,
    tokenizer_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    removal_method: str = "last_first",  # last_first or longest_first
) -> str:
    """
    Reads a prompt file, filters examples, and truncates to fit model context.

    This function is the primary tool for constructing a few-shot prompt. It
    parses a file where examples are separated by metadata lines, filters them
    based on criteria, and then intelligently removes examples until the total
    estimated token count is within the model's limit.

    Args:
        file_path: Path to the prompt file, or a list of paths.
        filter_by_key_values: A dictionary to filter examples. The key is the
            metadata field, and the value is a list of accepted values.
            e.g., {"skill": ["composition.1hop"]}
        metadata_prefix: The prefix string that identifies a metadata line.
        order_by_key: Key to order the final examples by.
        test_to_train_length_scale: A scaling factor to estimate the length of
            the unseen test example based on the training examples.
        estimated_generation_length: The estimated number of tokens the model
            will generate as a response. This is reserved in the context window.
        shuffle: Whether to shuffle the selected few-shot examples.
        model_length_limit: The total context window size of the target LLM.
        tokenizer_model_name: The tokenizer to use for calculating token length.
        removal_method: Strategy for removing examples if the prompt is too long.
            "last_first": Removes examples from the end of the file.
            "longest_first": Removes the longest examples first.

    Returns:
        A string containing the final, truncated prompt.
    """

    if not file_path:
        return ""

    # ---- 1. Read prompt file(s) into memory ----
    if isinstance(file_path, list):
        contents = []
        for _file_path in file_path:
            assert os.path.exists(_file_path), f"Filepath {_file_path} not available."
            with open(_file_path, "r") as file:
                content = file.read().strip()
                contents.append(content)
        content = "\n\n\n".join(contents)
        all_prompt_lines = [e + "\n" for e in content.strip().split("\n")]

    elif isinstance(file_path, str):
        assert os.path.exists(file_path), f"Filepath {file_path} not available."
        with open(file_path, "r") as file:
            all_prompt_lines = [e + "\n" for e in file.read().strip().split("\n")]

    else:
        raise Exception("Unexpected file_path type.")

    # ---- 2. Parse examples based on metadata prefix ----
    # Each example is a dictionary with metadata and the raw text lines.
    example = {"default": True, "lines": []}
    examples = []

    for index, line in enumerate(all_prompt_lines):

        if index == len(all_prompt_lines) - 1:
            # Add the last example
            examples.append(example)

        if line.strip().startswith(metadata_prefix):
            # A new example starts, so save the previous one.
            examples.append(example)
            metadata = line.strip().replace(metadata_prefix, "", 1)
            metadata = json.loads(metadata)
            example = copy.deepcopy(metadata)
            example["lines"] = []
        else:
            example["lines"].append(line)

    # ---- 3. Filter examples based on provided key-value pairs ----
    if filter_by_key_values:
        valid_examples = []
        for key, valid_values in filter_by_key_values.items():
            assert isinstance(valid_values, list)
            for example in examples:
                if not example["lines"]:
                    continue
                if key not in example:
                    print(f"WARNING: Key {key} not found in the prompt file_path ({file_path}). Skipping it.")
                    continue
                if example[key] in valid_values:
                    valid_examples.append(example)
        examples = valid_examples

    if order_by_key:
        examples = sorted(examples, key=lambda example: filter_by_key_values[key].index(example[key]))
        assert not shuffle

    prompt_examples_texts = ["".join(example["lines"]).strip() for example in examples]

    if len(prompt_examples_texts) == 1:
        # Nothing to compress. Return it as it is.
        prompt = prompt_examples_texts[0].strip()

    else:
        # ---- 4. Dynamically remove examples to fit the context window ----
        tokenizer = get_tokenizer(tokenizer_model_name)
        prompt_example_lengths = [len(tokenizer.tokenize(example_text)) for example_text in prompt_examples_texts]

        prompt_examples_original = len(prompt_examples_texts)
        prompt_examples_dropped = 0

        # This loop iteratively removes examples until the estimated total length fits.
        while prompt_example_lengths:

            # Estimate the total length required for the prompt.
            estimated_test_example_length = max(prompt_example_lengths) * test_to_train_length_scale
            estimated_total_length = (
                sum(prompt_example_lengths) + estimated_test_example_length + estimated_generation_length
            )

            if estimated_total_length > model_length_limit:
                # If too long, remove one example based on the chosen method.
                if removal_method == "longest_first":
                    max_length_index = prompt_example_lengths.index(max(prompt_example_lengths))
                    prompt_examples_texts.pop(max_length_index)
                    prompt_example_lengths.pop(max_length_index)
                    prompt_examples_dropped += 1

                elif removal_method == "last_first":
                    prompt_examples_texts.pop()
                    prompt_example_lengths.pop()
                    prompt_examples_dropped += 1

                else:
                    raise Exception(f"Unknown removal method: {removal_method}")

            else:
                # The prompt now fits, so we can exit the loop.
                break

        if not prompt_examples_texts:
            print("EXTREME WARNING: Not prompt examples remain.")

        if prompt_examples_dropped > 0:
            print(f"WARNING: Dropped {prompt_examples_dropped} / {prompt_examples_original} examples.")

        # ---- 5. Finalize the prompt string ----
        if shuffle:
            assert order_by_key is None
            random.shuffle(prompt_examples_texts)

        prompt = "\n\n\n".join([e.strip() for e in prompt_examples_texts])

    prompt = prompt.strip()
    return prompt


def fit_prompt_into_given_limit(
    original_prompt: str,
    model_length_limit: int,
    estimated_generation_length: int,
    demonstration_delimiter: str = "\n\n\n",
    shuffle: bool = False,
    remove_method: str = "first",  # first, last, random, largest
    tokenizer_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    last_is_test_example: bool = True,
):
    """
    Truncates a pre-constructed prompt string to fit a model's length limit.

    This is a more general-purpose utility than `read_prompt`. It takes a
    string, splits it into demonstrations, and removes them until the prompt
    fits. It's useful when the prompt is assembled programmatically, not just
    from a file.

    Args:
        original_prompt: The full prompt string with demonstrations.
        model_length_limit: The context window size of the target LLM.
        estimated_generation_length: Reserved space for the model's output.
        demonstration_delimiter: The string that separates few-shot examples.
        shuffle: Whether to shuffle the demonstrations before returning.
        remove_method: Strategy for removing demonstrations.
            "first": Remove from the beginning.
            "last": Remove from the end.
            "random": Remove randomly.
            "largest": Remove the longest one.
        tokenizer_model_name: Tokenizer for calculating length.
        last_is_test_example: If True, the last demonstration is treated as the
            final test query and is not removed (unless absolutely necessary).

    Returns:
        The truncated prompt string.
    """
    assert remove_method in (
        "first",
        "last",
        "random",
        "largest",
    ), "The remove_method must be from first, last, random, largest."

    # ---- 1. Split the prompt into individual demonstrations ----
    demonstrations = original_prompt.strip().split(demonstration_delimiter)
    demonstrations = [demonstration.strip() for demonstration in demonstrations if demonstration.strip()]

    if len(demonstrations) <= 1:
        print("EXTREME WARNING: Found only one demonstration/example.")

    tokenizer = get_tokenizer(tokenizer_model_name)
    demonstration_sizes = [len(tokenizer.tokenize(demonstration)) for demonstration in demonstrations]

    # ---- 2. Separate the test example (if applicable) ----
    # The test example is the final part of the prompt that we want to preserve.
    test_example = None
    test_example_size = 0
    if last_is_test_example:
        test_example = demonstrations.pop(-1)
        test_example_size = demonstration_sizes.pop(-1)

    # ---- 3. Iteratively remove demonstrations until the prompt fits ----
    while True:
        updated_length = sum(demonstration_sizes) + test_example_size + estimated_generation_length
        if updated_length < model_length_limit or not demonstration_sizes:
            break

        if remove_method == "first":
            remove_index = 0
        elif remove_method == "last":
            remove_index = -1
        elif remove_method == "random":
            remove_index = random.randint(0, len(demonstrations) - 1)
        elif remove_method == "largest":
            remove_index = demonstration_sizes.index(max(demonstration_sizes))
        else:
            raise Exception(f"Unexpected remove_method: {remove_method}.")

        demonstrations.pop(remove_index)
        demonstration_sizes.pop(remove_index)

        assert len(demonstrations) == len(demonstration_sizes)

    # ---- 4. Reconstruct the final prompt ----
    if shuffle:
        random.shuffle(demonstrations)

    if last_is_test_example:
        updated_prompt = demonstration_delimiter.join(demonstrations + [test_example])
    else:
        updated_prompt = demonstration_delimiter.join(demonstrations)

    # ---- 5. Final emergency truncation if still too long ----
    # This happens if the test example alone is too big for the context window.
    if updated_length > model_length_limit:
        print("EXTREME WARNING: Not enough space to fit the test example. Truncating it.")
        updated_lines = updated_prompt.split("\n")
        while updated_lines:
            # Remove lines from the beginning of the test example
            updated_lines.pop(0)
            if len(tokenizer.tokenize("\n".join(updated_lines))) <= model_length_limit:
                break
        updated_prompt = "\n".join(updated_lines)

    return updated_prompt
