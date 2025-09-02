#!/usr/bin/env python3
"""
Common Utilities for Prompt Generation in Adaptive RAG System

This module provides foundational utilities for generating experimental prompts
in few-shot learning scenarios. Implements sophisticated text processing,
experimental design patterns, and prompt formatting capabilities essential
for systematic evaluation of retrieval-augmented generation systems.

Core Scientific Capabilities:
1. Experimental Reproducibility
   - Fixed random seed for consistent experimental results
   - Robust sampling with automatic size adjustment
   - Systematic prompt organization and formatting

2. Advanced Text Processing
   - spaCy-based linguistic analysis with caching
   - Sentence-aware text truncation preserving semantic coherence
   - Context length management with content prioritization

3. Prompt Generation Framework
   - Abstract base classes implementing Template Method pattern
   - Task-specific prompt generators for QA and retrieval
   - Configurable demonstration formatting for different models

4. Multi-Modal Experimental Design
   - Support for direct answer generation and chain-of-thought reasoning
   - Controlled distractor injection for robustness evaluation
   - Context provision strategies (no context vs. gold+distractors)

The module follows established practices in few-shot learning research,
implementing systematic experimental designs that enable fair comparison
across different model architectures and reasoning paradigms.
"""

from typing import List, Dict, Tuple, Union, Any
from functools import lru_cache
import json
import random
import copy
import re

# Fixed random seed for experimental reproducibility
# Critical for ensuring consistent experimental results across runs
# Seed value (13370) chosen to be non-standard but memorable
random.seed(13370)  # Don't change - essential for reproducibility


def safe_sample(items: List[Any], count: int) -> List[Any]:
    """
    Perform safe random sampling with automatic size adjustment.
    
    This utility function implements robust sampling that handles edge cases
    where the requested sample size exceeds the available population size.
    Essential for experimental robustness in few-shot learning scenarios.
    
    Scientific Rationale:
    In few-shot learning experiments, available examples may be limited.
    This function ensures experiments can proceed even when sample sizes
    exceed population sizes, maintaining experimental validity.
    
    Args:
        items (List[Any]): Population to sample from
        count (int): Desired sample size
        
    Returns:
        List[Any]: Random sample of size min(count, len(items))
        
    Algorithm:
        1. Compute feasible sample size as min(requested, available)
        2. If feasible size > 0, perform random sampling without replacement
        3. If feasible size = 0, return empty list
    """
    # Ensure sample size doesn't exceed population size
    count = min(count, len(items))
    # Perform random sampling without replacement, handling empty case
    return random.sample(items, count) if count > 0 else []


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read and parse JSON Lines format files for experimental data loading.
    
    JSON Lines (JSONL) format is standard in NLP research for storing
    structured experimental data. Each line contains a complete JSON object,
    enabling efficient streaming processing of large datasets.
    
    Scientific Application:
    - Loads experimental instances for few-shot learning
    - Enables efficient processing of large-scale NLP datasets
    - Maintains data integrity through structured parsing
    
    Args:
        file_path (str): Path to JSONL file containing experimental data
        
    Returns:
        List[Dict]: List of parsed JSON objects representing experimental instances
        
    Error Handling:
        - Skips empty lines to handle file formatting inconsistencies
        - Maintains data integrity through JSON parsing validation
    """
    with open(file_path, "r") as file:
        # Parse each non-empty line as a JSON object
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


@lru_cache(maxsize=None)
def get_spacy_object():
    """
    Initialize and cache spaCy NLP pipeline for text processing.
    
    spaCy provides industrial-strength natural language processing capabilities
    including tokenization, sentence segmentation, and linguistic analysis.
    LRU caching ensures single initialization across multiple function calls.
    
    Scientific Rationale:
    - Consistent tokenization across all text processing operations
    - Efficient sentence boundary detection for paragraph clipping
    - Cached initialization prevents redundant model loading
    
    Returns:
        spacy.Language: Cached spaCy English language model
        
    Model Selection:
    Uses 'en_core_web_sm' - a compact English model optimized for:
    - Fast processing speed suitable for large-scale experiments
    - Accurate sentence segmentation for paragraph truncation
    - Consistent tokenization for length calculations
    """
    import spacy
    # Load compact English model optimized for efficiency
    return spacy.load("en_core_web_sm")


def clip_paragraph_text(paragraph_text: str, max_tokens: int = 250) -> str:
    """
    Intelligently truncate paragraph text while preserving sentence boundaries.
    
    This function implements sentence-aware text truncation, ensuring that
    clipped text maintains semantic coherence by preserving complete sentences.
    Critical for maintaining context quality in few-shot learning scenarios.
    
    Scientific Motivation:
    - Preserves semantic coherence through sentence-boundary awareness
    - Maintains consistent context lengths across experimental conditions
    - Prevents mid-sentence truncation that could confuse language models
    
    Algorithm:
    1. Parse text into sentences using spaCy's sentence segmentation
    2. Iteratively add complete sentences until token limit approached
    3. Stop before exceeding limit to maintain semantic integrity
    4. Preserve whitespace formatting for natural text flow
    
    Args:
        paragraph_text (str): Original paragraph text to truncate
        max_tokens (int): Maximum token count (default: 250)
        
    Returns:
        str: Truncated text preserving complete sentences
        
    Tokenization:
    Uses spaCy's tokenization which aligns with modern NLP practices
    and provides consistent results across different text types.
    """
    # Get cached spaCy pipeline for consistent processing
    spacy_object = get_spacy_object()
    # Parse text into linguistic objects for sentence segmentation
    paragraph_object = spacy_object(paragraph_text)
    # Extract sentence boundaries using spaCy's sentence segmentation
    paragraph_sents = paragraph_object.sents
    
    # Initialize counters for iterative sentence addition
    clipped_paragraph_tokens = 0
    clipped_paragraph_text = ""
    
    # Iteratively add complete sentences until token limit approached
    for sent in paragraph_sents:
        # Check if adding this sentence would exceed token limit
        if clipped_paragraph_tokens + len(sent) >= max_tokens:
            break
        # Add sentence with preserved whitespace formatting
        clipped_paragraph_text += sent.text_with_ws
        # Update token count using spaCy's tokenization
        clipped_paragraph_tokens += len(sent)
    
    return clipped_paragraph_text


def clip_paragraphs(paragraphs: List[Dict], max_tokens: int = 250):
    """
    Apply intelligent text clipping to collections of paragraphs with preservation rules.
    
    This function implements selective paragraph clipping that preserves important
    context while truncating less critical information. Supports experimental
    designs requiring controlled context lengths with content prioritization.
    
    Scientific Design:
    - Preserves supporting evidence paragraphs in their entirety
    - Preserves pinned context that provides essential background
    - Selectively truncates non-essential context to maintain focus
    - Maintains consistent paragraph structure for downstream processing
    
    Preservation Rules:
    1. Supporting paragraphs: Contain evidence directly relevant to answers
    2. Pinned paragraphs: Provide essential background context
    3. Regular paragraphs: Subject to truncation for length management
    
    Args:
        paragraphs (List[Dict]): List of paragraph dictionaries with metadata
        max_tokens (int): Maximum tokens per paragraph (default: 250)
        
    Returns:
        List[Dict]: Deep copy of paragraphs with selective clipping applied
        
    Paragraph Structure:
    Each paragraph dictionary contains:
    - paragraph_text: The actual text content
    - is_supporting: Boolean indicating evidential importance
    - is_pinned: Boolean indicating essential background status
    - Additional metadata preserved unchanged
    """
    # Create deep copy to avoid modifying original data structures
    paragraphs = copy.deepcopy(paragraphs)
    
    # Apply selective clipping based on paragraph importance
    for paragraph in paragraphs:
        # Preserve supporting evidence and pinned context in full
        if paragraph["is_supporting"] or paragraph["is_pinned"]:
            continue
        
        # Extract text content for processing
        paragraph_text = paragraph["paragraph_text"]
        # Apply sentence-aware clipping to non-essential paragraphs
        clipped_paragraph_text = clip_paragraph_text(paragraph_text, max_tokens)
        # Update paragraph with clipped content
        paragraph["paragraph_text"] = clipped_paragraph_text
    
    return paragraphs


class PromptGenerator:
    """
    Abstract base class for systematic prompt generation in few-shot learning experiments.
    
    This class provides the foundational architecture for generating experimental prompts
    across different tasks and model architectures. Implements standardized prompt
    formatting and demonstration organization following established practices in
    few-shot learning research.
    
    Scientific Foundation:
    - Standardized prompt formatting for consistent experimental conditions
    - Flexible demonstration organization supporting various experimental designs
    - Metadata preservation for experimental traceability and reproducibility
    - Delimiter-based prompt structure following few-shot learning conventions
    
    Design Pattern:
    Implements the Template Method pattern where subclasses define task-specific
    prompt generation while this base class handles common formatting and organization.
    
    Experimental Features:
    - Metadata injection for experimental tracking
    - Configurable demonstration delimiters for model-specific formatting
    - Optional demonstration sampling for training data efficiency
    - Systematic prompt organization for large-scale experiments
    """
    
    def __init__(
        self,
        input_file_path: str,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        """
        Initialize prompt generator with experimental configuration.
        
        Args:
            input_file_path (str): Path to JSONL file containing training instances
            demonstration_delimiter (str): String separating demonstrations in prompts
            one_demonstration_per_instance (bool): Whether to sample single demonstration per instance
        """
        # Load experimental instances from structured data file
        self._instances = read_jsonl(input_file_path)
        # Configure demonstration formatting for model-specific requirements
        self._demonstration_delimiter = demonstration_delimiter
        # Configure sampling strategy for demonstration selection
        self._one_demonstration_per_instance = one_demonstration_per_instance

    def _generate(self, instance: Dict) -> str:
        """
        Abstract method for task-specific prompt generation.
        
        Subclasses must implement this method to define how individual instances
        are converted into demonstration format for their specific task.
        
        Args:
            instance (Dict): Single experimental instance with task-specific structure
            
        Returns:
            str: Generated demonstration(s) for this instance
            
        Raises:
            NotImplementedError: This is an abstract method requiring implementation
        """
        raise NotImplementedError

    def generate(self) -> str:
        """
        Generate complete prompt containing all demonstrations with experimental metadata.
        
        This method orchestrates the complete prompt generation process, including
        metadata injection, demonstration generation, and final formatting.
        Ensures experimental reproducibility through systematic organization.
        
        Returns:
            str: Complete prompt ready for model evaluation
            
        Process:
        1. Generate demonstrations for each instance using task-specific logic
        2. Inject experimental metadata for traceability
        3. Apply demonstration sampling if configured
        4. Combine demonstrations with appropriate delimiters
        5. Return formatted prompt for model consumption
        
        Metadata Format:
        Each demonstration includes JSON metadata with question_id for:
        - Experimental result tracking
        - Performance analysis by instance
        - Reproducibility verification
        """
        def instance_to_header(instance):
            """Generate experimental metadata header for traceability."""
            return "# METADATA: " + json.dumps({"qid": instance["question_id"]})

        # Generate demonstrations with experimental metadata
        all_demonstrations_with_headers = []
        for instance in self._instances:
            # Generate task-specific demonstrations for this instance
            local_demonstrations_with_headers = [
                "\n".join([instance_to_header(instance), demonstration]).strip()
                for demonstration in self._generate(instance)
            ]
            
            # Apply demonstration sampling if configured
            if len(local_demonstrations_with_headers) > 1 and self._one_demonstration_per_instance:
                # Randomly sample single demonstration for efficiency
                all_demonstrations_with_headers.append(random.choice(local_demonstrations_with_headers))
            else:
                # Include all demonstrations for comprehensive evaluation
                all_demonstrations_with_headers += local_demonstrations_with_headers

        # Combine demonstrations with configured delimiter
        generated_output = self._demonstration_delimiter.join(all_demonstrations_with_headers)
        return generated_output


class QAPromptGenerator(PromptGenerator):
    """
    Specialized prompt generator for Question Answering tasks with multi-hop reasoning.
    
    This class implements sophisticated prompt generation for QA tasks requiring
    multi-step reasoning over multiple information sources. Supports both direct
    answer generation and chain-of-thought reasoning paradigms across different
    context configurations.
    
    Scientific Capabilities:
    - Multi-hop reasoning support through context assembly
    - Chain-of-thought reasoning with explicit step documentation
    - Controlled distractor injection for robustness evaluation
    - Model-specific prompt formatting for optimal performance
    - Context prioritization through pinning mechanisms
    
    Experimental Design Features:
    - Factorial experimental design across reasoning types and context levels
    - Systematic distractor injection for noise robustness testing
    - Context length management for consistent experimental conditions
    - Model-specific optimizations for Codex and FLAN-T5 architectures
    """
    
    def __init__(
        self,
        input_file_path: str,
        qa_type: str,
        context_type: str,
        model_name: str,
        distractor_count: Union[int, Tuple[int, int]] = 0,
        max_paragraph_tokens: int = 250,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
        pinned_at_bottom: bool = True,
    ):
        """
        Initialize QA prompt generator with comprehensive experimental configuration.
        
        Args:
            input_file_path (str): Path to annotated training data
            qa_type (str): Reasoning paradigm ("direct" or "cot")
            context_type (str): Context provision strategy ("no" or "gold_with_distractors")
            model_name (str): Target model architecture ("codex", "flan_t5", "gemini", or "qwen")
            distractor_count (Union[int, Tuple[int, int]]): Number/range of distractors
            max_paragraph_tokens (int): Token limit for paragraph truncation
            demonstration_delimiter (str): String separating demonstrations
            one_demonstration_per_instance (bool): Single demonstration sampling flag
            pinned_at_bottom (bool): Whether to place pinned context at bottom
        """
        # Initialize base prompt generator functionality
        super().__init__(input_file_path, demonstration_delimiter, one_demonstration_per_instance)
        
        # Validate and store reasoning paradigm configuration
        assert qa_type in ("direct", "cot"), f"qa_type must be in direct, cot. Found {qa_type}."
        self._qa_type = qa_type
        
        # Validate and store context provision strategy
        assert context_type in (
            "no",
            "gold_with_distractors",
        ), f"context_type must be in no or gold_with_distractors. Found {context_type}."
        self._context_type = context_type
        
        # Enforce logical constraints between context type and distractors
        if context_type == "no":
            assert distractor_count == 0
        
        # Support both fixed and variable distractor counts for experimental flexibility
        assert isinstance(distractor_count, int) or isinstance(distractor_count, tuple)
        self._distractor_count = distractor_count
        
        # Store paragraph length management configuration
        self._max_paragraph_tokens = max_paragraph_tokens
        
        # Validate and store model-specific configuration
        assert model_name in ("codex", "flan_t5", "gemini", "qwen"), f"model_name must be in codex, flan_t5, gemini, qwen. Found {model_name}."
        self._model_name = model_name
        
        # Store context organization preference
        assert pinned_at_bottom in (True, False), "pinned_at_bottom must be True or False."
        self._pinned_at_bottom = pinned_at_bottom

    def _generate(self, instance: Dict) -> List[Dict]:
        """
        Generate QA demonstration with multi-hop reasoning context assembly.
        
        This method implements sophisticated context assembly for multi-hop reasoning
        tasks, combining gold evidence with controlled distractor injection and
        chain-of-thought reasoning generation.
        
        Args:
            instance (Dict): Single QA instance with reasoning steps and contexts
            
        Returns:
            List[Dict]: List containing single demonstration string
            
        Algorithm:
        1. Extract and organize gold evidence from reasoning steps
        2. Generate chain-of-thought reasoning if configured
        3. Inject controlled distractors for robustness testing
        4. Assemble context with appropriate prioritization
        5. Format demonstration according to model requirements
        6. Apply length constraints while preserving semantic integrity
        """
        # Initialize chain-of-thought reasoning collection
        cot_sents = []
        reasoning_steps = instance["reasoning_steps"]

        # Phase 1: Gather Gold Evidence Paragraphs and Generate CoT
        # Initialize gold evidence collection with pinned contexts
        gold_paragraphs = instance.get("pinned_contexts", [])
        # Track processed paragraphs to prevent duplication
        taken_gold_title_paras = []
        
        # Process pinned contexts (essential background information)
        for paragraph in gold_paragraphs:
            paragraph["is_pinned"] = True        # Mark as essential context
            paragraph["is_supporting"] = True    # Mark as supporting evidence
            # Create unique identifier for deduplication
            title_para = (paragraph["title"], paragraph["paragraph_text"])
            taken_gold_title_paras.append(title_para)

        # Process reasoning steps to extract evidence and generate CoT
        for reasoning_step in reasoning_steps:
            # Extract chain-of-thought sentence if CoT reasoning is enabled
            if self._qa_type == "cot":
                cot_sents.append(reasoning_step["cot_sent"])

            # Extract supporting paragraph for this reasoning step
            assert len(reasoning_step["paragraphs"]) == 1
            paragraph = reasoning_step["paragraphs"][0]

            # Skip paragraphs without title (incomplete evidence)
            if not paragraph["title"]:
                continue

            # Check for duplicate evidence to prevent redundancy
            title_para = (paragraph["title"], paragraph["paragraph_text"])
            if title_para in taken_gold_title_paras:
                continue

            # Mark paragraph as supporting evidence
            paragraph["is_supporting"] = True
            paragraph["is_pinned"] = False

            # Add to gold evidence collection
            gold_paragraphs.append(paragraph)
            taken_gold_title_paras.append(title_para)

        # Apply context filtering based on experimental condition
        if self._context_type == "no":
            # No-context condition: only retain pinned contexts
            gold_paragraphs = [paragraph for paragraph in gold_paragraphs if paragraph["is_pinned"]]

        # Phase 2: Gather Distractor Paragraphs for Robustness Testing
        distractor_paragraphs = []

        if self._context_type == "gold_with_distractors":
            # Determine distractor count (support both fixed and variable)
            if isinstance(self._distractor_count, int):
                distractor_count = self._distractor_count
            else:
                # Sample from range for experimental variability
                distractor_count = random.randint(*self._distractor_count)
            
            # Identify candidate distractor paragraphs
            candidate_distractor_paragraphs = [
                paragraph for paragraph in instance["contexts"] if not paragraph["is_supporting"]
            ]
            # Remove paragraphs already used as gold evidence
            candidate_distractor_paragraphs = [
                paragraph
                for paragraph in candidate_distractor_paragraphs
                if (paragraph["title"], paragraph["paragraph_text"]) not in taken_gold_title_paras
            ]
            
            # Mark distractor paragraphs appropriately
            for paragraph in candidate_distractor_paragraphs:
                assert not paragraph.get("is_supporting", False)
                paragraph["is_supporting"] = False
                paragraph["is_pinned"] = False
            
            # Randomly sample distractors for controlled noise injection
            distractor_paragraphs = safe_sample(candidate_distractor_paragraphs, distractor_count)

        # Phase 3: Assemble Complete Context with Intelligent Organization
        all_paragraphs = gold_paragraphs + distractor_paragraphs
        # Apply length constraints while preserving important content
        all_paragraphs = clip_paragraphs(all_paragraphs, max_tokens=self._max_paragraph_tokens)

        # Randomize paragraph order to prevent position bias
        random.shuffle(all_paragraphs)
        # Apply context prioritization through pinning strategy
        all_paragraphs = sorted(all_paragraphs, key=lambda e: int(e["is_pinned"]), reverse=not self._pinned_at_bottom)

        # Format context text with consistent Wikipedia-style formatting
        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in all_paragraphs
            ]
        ).strip()

        # Phase 4: Prepare Answer and Chain-of-Thought Components
        # Extract gold standard answer (assumes single answer format)
        assert len(instance["answers_objects"]) == 1
        assert len(instance["answers_objects"][0]["spans"]) == 1
        answer_text = instance["answers_objects"][0]["spans"][0]

        # Prepare chain-of-thought reasoning text
        cot_text = re.sub(r" +", " ", " ".join(cot_sents))

        # Phase 5: Generate Final Demonstration with Model-Specific Formatting
        # Configure output based on reasoning paradigm
        if self._qa_type == "direct":
            output_text = "A: " + answer_text
            qn_pretext = "Q:"
        elif self._qa_type == "cot":
            output_text = "A: " + cot_text
            qn_pretext = "Q:"
        else:
            raise Exception(f"Encountered unknown choice of qa_type {self._qa_type}.")

        # Apply model-specific question instruction formatting
        question_instruction = ""
        if self._model_name.startswith(("flan", "gemini", "qwen")) and self._qa_type == "direct":
            question_instruction = "Answer the following question.\n"
        elif self._model_name.startswith(("flan", "gemini", "qwen")) and "cot" in self._qa_type:
            question_instruction = "Answer the following question by reasoning step-by-step.\n"

        # Assemble complete demonstration with proper formatting
        question_text = instance["question_text"]
        demonstration = "\n".join(
            [context_text, "", f"{qn_pretext} {question_instruction}{question_text}", output_text]
        ).strip()

        return [demonstration]


class NoContextOpenRetrieverPromptGenerator(PromptGenerator):
    """
    Specialized prompt generator for open-domain retrieval tasks without contextual grounding.
    
    This class implements prompt generation for evaluating language models' ability
    to perform information retrieval using only their parametric knowledge. Tests
    whether models can identify relevant information sources without explicit context,
    representing a fundamental challenge in knowledge-intensive NLP tasks.
    
    Scientific Motivation:
    Open-domain retrieval requires models to leverage their pre-training knowledge
    to identify relevant information sources. This capability is crucial for:
    - Knowledge-intensive question answering systems
    - Information seeking in large-scale knowledge bases
    - Evaluation of parametric knowledge in large language models
    
    Experimental Design:
    - Zero-shot retrieval evaluation (no example retrievals provided)
    - Cross-architecture comparison (Codex vs FLAN-T5)
    - Systematic evaluation of retrieval accuracy and relevance
    - Preservation of reasoning step ordering for learning signal
    """
    
    def __init__(
        self,
        input_file_path: str,
        model_name: str,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        """
        Initialize open-domain retrieval prompt generator.
        
        Args:
            input_file_path (str): Path to annotated training data
            model_name (str): Target model architecture ("codex", "flan_t5", "gemini", or "qwen")
            demonstration_delimiter (str): String separating demonstrations
            one_demonstration_per_instance (bool): Single demonstration sampling flag
        """
        # Initialize base prompt generator functionality
        super().__init__(input_file_path, demonstration_delimiter, one_demonstration_per_instance)
        
        # Validate and store model architecture configuration
        assert model_name in ("codex", "flan_t5", "gemini", "qwen"), f"model_name must be in codex, flan_t5, gemini, qwen. Found {model_name}."
        self._model_name = model_name

    def _generate(self, instance: Dict) -> List[Dict]:
        """
        Generate open-domain retrieval demonstration with parametric knowledge requirements.
        
        This method creates demonstrations that test models' ability to identify
        relevant information sources using only their parametric knowledge, without
        explicit contextual grounding or retrieval examples.
        
        Args:
            instance (Dict): Single retrieval instance with question and gold titles
            
        Returns:
            List[Dict]: List containing single demonstration string
            
        Algorithm:
        1. Extract question and any pinned context
        2. Identify gold standard retrieval targets from reasoning steps
        3. Preserve ordering of retrieval targets for learning signal
        4. Format demonstration according to model requirements
        5. Generate appropriate output format (list vs comma-separated)
        
        Design Principles:
        - Maintains reasoning step ordering for enhanced learning
        - Excludes pinned contexts from retrieval targets (already provided)
        - Handles model-specific formatting requirements
        - Preserves experimental validity through systematic target identification
        """
        # Extract question text for retrieval task
        question_text = instance["question_text"]

        # Process pinned contexts (background information provided to model)
        # NOTE: pinned_contexts are added by default. Can make it configurable later if needed.
        pinned_paragraphs = instance.get("pinned_contexts", [])
        
        # Format pinned contexts with consistent Wikipedia-style formatting
        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in pinned_paragraphs
            ]
        ).strip()

        # Extract gold standard retrieval targets from reasoning steps
        # NOTE: Make sure to retain the order of the paragraph titles.
        # Since they follow the reasoning steps, the order might help learn the task better.
        gold_titles = []
        
        # Create set of already-provided titles to avoid duplication
        avoid_gold_titles = set(paragraph["title"] for paragraph in instance.get("pinned_contexts", []))
        
        # Process reasoning steps to identify retrieval targets
        for reasoning_step in instance["reasoning_steps"]:
            # Extract title from reasoning step paragraph
            assert len(reasoning_step["paragraphs"]) == 1
            title = reasoning_step["paragraphs"][0]["title"]
            
            # Skip empty titles (incomplete reasoning steps)
            if not title:
                continue
            
            # Skip duplicate titles and already-provided contexts
            if title in gold_titles or title in avoid_gold_titles:
                continue
            
            # Add to gold standard retrieval targets
            gold_titles.append(reasoning_step["paragraphs"][0]["title"])

        # Determine retrieval count for task specification
        retrieval_count = len(gold_titles)

        # Apply model-specific question formatting
        question_prefix = ""
        if self._model_name.startswith(("flan", "gemini", "qwen")):
            question_prefix = "Answer the following question.\n"

        # Construct retrieval task specification
        input_text = (
            f"{question_prefix}The question is: '{question_text}'. "
            f"Generate titles of {retrieval_count} Wikipedia pages that have relevant information to answer this question."
        )

        # Format output according to model requirements
        if self._model_name.startswith(("flan", "gemini", "qwen")):
            # Instruction-tuned models prefer comma-separated natural language format
            output_text = ", ".join(gold_titles).strip()
        else:
            # Codex prefers structured list format with proper quoting
            output_text = "[" + ", ".join(['"' + e.replace('"', "'") + '"' for e in gold_titles]) + "]"

        # Assemble complete demonstration with proper formatting
        demonstration = "\n".join([context_text, "", f"Q: {input_text}", f"A: {output_text}"]).strip()

        return [demonstration]
