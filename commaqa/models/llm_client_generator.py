"""
LLM Client for Remote Generation Server


Key components:
- `LLMClientGenerator`: A high-level class that simplifies text generation. It
  handles prompt truncation to fit within the model's context window and
  parses the server's response.
- `llm_call`: The main function for making API calls. It acts as a dispatcher,
  deciding whether to use a cached result or make a live API call based on
  the generation parameters (sampling vs. deterministic).
- No caching: All calls are made directly to the server for real-time results.

The client discovers the LLM server host and port via environment variables,
allowing for flexible deployment configurations (e.g., running different model
servers on different ports).
"""
import requests
import os
from typing import Dict

from commaqa.inference.prompt_reader import fit_prompt_into_given_limit


def non_cached_llm_call(  # kwargs doesn't work with caching.
    prompt: str,
    model_name: str,
    max_input: int = None,
    max_length: int = 100,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    keep_prompt: bool = False,
) -> Dict:
    """
    Makes a live, non-cached request to the LLM server.

    It constructs the server URL from environment variables, sends the generation
    request, and validates the response.

    Args:
        prompt: The input text to the model.
        model_name: The name of the model to use for generation.
        ... (other generation parameters)

    Returns:
        A dictionary containing the server's JSON response.

    Raises:
        Exception: If the request fails or the responding model is not the
                   one that was requested.
    """
    params = {
        "prompt": prompt,
        "max_input": max_input,
        "max_length": max_length,
        "min_length": min_length,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "keep_prompt": keep_prompt,
    }

    # Discover server host and port from environment variables.
    # Allows for a default server and model-specific overrides.
    host = os.environ.get("LLM_SERVER_HOST", None)
    port = os.environ.get("LLM_SERVER_PORT", None)

    model_name_simple = model_name
    if "/" in model_name:
        assert model_name.count("/", 1)
        model_name_simple = model_name.split("/")[1]

    llm_server_key_suffix = os.environ.get("LLM_SERVER_KEY_SUFFIX", "")
    # Check for model-specific host/port env vars
    env_var_prefix = model_name_simple.replace("-", "_")
    if env_var_prefix + "_LLM_SERVER_HOST" + llm_server_key_suffix in os.environ:
        host = os.environ[env_var_prefix + "_LLM_SERVER_HOST" + llm_server_key_suffix]
    if env_var_prefix + "_LLM_SERVER_PORT" + llm_server_key_suffix in os.environ:
        port = os.environ[env_var_prefix + "_LLM_SERVER_PORT" + llm_server_key_suffix]

    response = requests.get(f"http://{host}:{port}/generate", params=params)

    if response.status_code != 200:
        raise Exception(f"LLM Generation request failed with status {response.status_code}: {response.text}")

    result = response.json()

    # Validate that the response came from the correct model.
    # This is a safeguard against connecting to a misconfigured server.
    response_model_name = result.get("model_name", "")
    # Normalize names by removing quantization suffixes for comparison.
    clean_response_model = response_model_name.replace("-bf16", "").replace("-dsbf16", "").replace("-8bit", "")
    
    # Extract simple name from server response for comparison (handle both full and simple formats)
    clean_response_simple = clean_response_model
    if "/" in clean_response_model:
        clean_response_simple = clean_response_model.split("/")[1]
    
    # Compare simplified names
    if clean_response_simple != model_name_simple:
        raise Exception(f"Looks like incorrect LLM server is ON: {response_model_name} != {model_name}.")

    return result


def cached_llm_call(  # kwargs doesn't work with caching.
    prompt: str,
    model_name: str,
    max_input: int = None,
    max_length: int = 100,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    keep_prompt: bool = False,
) -> Dict:
    """
    A wrapper around `non_cached_llm_call` (caching removed).

    This function now directly calls the non-cached version.
    Previously used caching for deterministic generation, but caching has been disabled.
    """
    return non_cached_llm_call(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


def llm_call(
    prompt: str,
    model_name: str,
    max_input: int = None,
    max_length: int = 100,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    keep_prompt: bool = False,
) -> Dict:
    """
    Dispatches to either the cached or non-cached LLM call function.

    Note: Caching has been disabled. All calls now go directly to the server
    regardless of generation parameters.
    """
    # Caching has been disabled - all calls now go directly to the server
    # Previously cached deterministic calls (when not sampling)
    function = non_cached_llm_call
    return function(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


class LLMClientGenerator:
    """
    A high-level client for generating text via the LLM server.

    This class wraps the `llm_call` function, providing a convenient interface
    that automatically handles prompt truncation to fit the model's context
    window before making the generation call.
    """
    


    def __init__(
        self,
        model_name: str,
        max_input: int = None,
        max_length: int = 100,
        min_length: int = 1,
        do_sample: bool = False,
        eos_text: str = "\n",
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        repetition_penalty: float = None,
        length_penalty: float = None,
        model_tokens_limit: int = 2048,  # FLAN-T5-Large context window
        remove_method: str = "first",
    ):
        """
        Initializes the LLM client with generation parameters.

        Args:
            model_name: The name of the model, e.g., "flan-t5-large".
            ... (other generation and truncation parameters)
        """
        # This list should be updated with the `model_name` portion (part after '/')
        # of any new models supported by the remote LLM server.
        # Currently only supporting FLAN-T5-Large for encoder-decoder classifier architecture
        valid_model_names = [
            "flan-t5-large",
        ]
        model_name_ = model_name
        if "/" in model_name:
            assert model_name.count("/", 1)
            model_name_ = model_name.split("/")[1]
        
        assert model_name_ in valid_model_names, f"Model name '{model_name}' (parsed as '{model_name_}') not in {valid_model_names}"

        self.model_name = model_name
        self.max_input = max_input
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_text = eos_text
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.model_tokens_limit = model_tokens_limit
        self.remove_method = remove_method


    def generate_text_sequence(self, prompt: str) -> list[tuple[str, float]]:
        """
        Generates text by processing a prompt and calling the LLM server.

        This method first truncates the prompt to ensure it fits within the
        model's context limit, then calls the LLM server, and finally
        parses the output to clean it up.

        Args:
            prompt: The full input prompt string.

        Returns:
            A list of (text, score) tuples, sorted by score.
        """
        prompt = prompt.rstrip()

        # Ensure the prompt fits the model's context window.
        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_length,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name=self.model_name,
            last_is_test_example=True,
        )

        # Note: Don't pass eos_text to the server. The client handles it manually.
        params = {
            "prompt": prompt,
            "model_name": self.model_name,
            "max_input": self.max_input,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "keep_prompt": False,
        }
        result = llm_call(**params)

        generated_texts = result["generated_texts"]
        modified_texts = []
        
        # Post-process the generated text to remove the prompt and stop at EOS.
        for text in generated_texts:
            if text.startswith(prompt):
                text = text[len(prompt) :]
            if self.eos_text and self.eos_text in text:
                text = text[: text.index(self.eos_text)]
            modified_texts.append(text)
        generated_texts = modified_texts

        # Assign a score to each generated sequence.
        output_seq_score = [(text, 1 / (index + 1)) for index, text in enumerate(generated_texts)]

        # TODO: Deal with output-probabilities if needed.

        return sorted(output_seq_score, key=lambda x: x[1])
