#!/usr/bin/env python3
"""
IRCOT Example Bank - Manages few-shot examples for prompting.

This module provides:
- High-quality multi-hop reasoning examples
- Dataset-specific examples
- Hop count categorization
- Unanswerable examples
"""

from typing import Dict, Any, List, Optional
import random
from collections import defaultdict


class IRCoTExampleBank:
    """
    Manages a collection of high-quality IRCOT examples.
    
    Examples are categorized by:
    - Number of hops (1, 2, 3, 4+)
    - Dataset origin
    - Answerability
    """
    
    def __init__(self, dataset: Optional[str] = None):
        """
        Initialize the example bank.
        
        Args:
            dataset: Optional dataset to filter examples
        """
        self.dataset = dataset
        self.examples = self._load_examples()
    
    def _load_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all examples organized by hop count."""
        examples = defaultdict(list)
        
        # 2-hop examples
        examples[2].extend([
            {
                "question": "When was Neville A. Stanton's employer founded?",
                "documents": [
                    {
                        "title": "Neville A. Stanton",
                        "text": "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton."
                    },
                    {
                        "title": "University of Southampton", 
                        "text": "The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students."
                    }
                ],
                "reasoning_steps": [
                    "I need to identify who Neville A. Stanton's employer is first.",
                    "From the previous information, I found that Neville A. Stanton works at the University of Southampton. The new context confirms that the University of Southampton was founded in 1862. So the answer is: 1862."
                ],
                "answer": "1862"
            },
            {
                "question": "What is the nationality of the founder of SolarTech Innovations?",
                "documents": [
                    {
                        "title": "SolarTech Innovations",
                        "text": "SolarTech Innovations is a renewable energy company that develops advanced solar panel technology."
                    },
                    {
                        "title": "Dr. Yuki Tanaka Profile",
                        "text": "SolarTech Innovations was founded by Dr. Yuki Tanaka in 2010. Dr. Tanaka is a Japanese engineer who previously worked at several major technology companies."
                    }
                ],
                "reasoning_steps": [
                    "I need to identify who founded SolarTech Innovations first.",
                    "Perfect! I found both pieces needed: SolarTech Innovations was founded by Dr. Yuki Tanaka, and Dr. Tanaka is Japanese. So the answer is: Japanese."
                ],
                "answer": "Japanese"
            },
            {
                "question": "What year did the lead actor of 'The Matrix' win his first major film award?",
                "documents": [
                    {
                        "title": "The Matrix",
                        "text": "The Matrix is a 1999 science fiction action film starring Keanu Reeves as Neo, a computer programmer who discovers the world is a simulation."
                    },
                    {
                        "title": "Keanu Reeves",
                        "text": "Keanu Reeves is a Canadian actor. He won his first major film award, the MTV Movie Award for Most Desirable Male, in 1992 for his role in Point Break."
                    }
                ],
                "reasoning_steps": [
                    "I need to identify the lead actor of 'The Matrix' first.",
                    "The Matrix starred Keanu Reeves as the lead. Now I found that Keanu Reeves won his first major film award in 1992. So the answer is: 1992."
                ],
                "answer": "1992"
            }
        ])
        
        # 3-hop examples
        examples[3].extend([
            {
                "question": "What year was the spouse of the director of Quantum Horizons born?",
                "documents": [
                    {
                        "title": "Quantum Horizons",
                        "text": "Quantum Horizons is a 2022 science fiction film that explores themes of time travel and alternate realities."
                    },
                    {
                        "title": "Elena Martinez Director Profile",
                        "text": "Quantum Horizons was directed by Elena Martinez, an acclaimed filmmaker known for her work in the science fiction genre."
                    },
                    {
                        "title": "Elena Martinez Personal Life",
                        "text": "Elena Martinez is married to composer David Kim, whom she met while working on a documentary film."
                    },
                    {
                        "title": "David Kim Composer Biography",
                        "text": "David Kim (born 1985) is a South Korean-American composer specializing in film scores."
                    }
                ],
                "reasoning_steps": [
                    "I need to find out who directed Quantum Horizons first.",
                    "Found the director: Elena Martinez. Now I need to find information about Elena Martinez's spouse.",
                    "Great! Now I know Elena Martinez is married to David Kim. I need David Kim's birth year to complete the chain.",
                    "Excellent! I now have the complete chain: Quantum Horizons → directed by Elena Martinez → married to David Kim → born 1985. So the answer is: 1985."
                ],
                "answer": "1985"
            },
            {
                "question": "In which city is the headquarters of the company that acquired DataFlow Inc. located?",
                "documents": [
                    {
                        "title": "DataFlow Inc.",
                        "text": "DataFlow Inc. was a data analytics startup that specialized in real-time processing solutions."
                    },
                    {
                        "title": "TechGiant Corporation Acquisitions",
                        "text": "TechGiant Corporation acquired DataFlow Inc. in 2021 as part of its expansion into the analytics market."
                    },
                    {
                        "title": "TechGiant Corporation",
                        "text": "TechGiant Corporation is a multinational technology company headquartered in Seattle, Washington."
                    }
                ],
                "reasoning_steps": [
                    "I need to find which company acquired DataFlow Inc. first.",
                    "Found that TechGiant Corporation acquired DataFlow Inc. Now I need to find where TechGiant Corporation is headquartered.",
                    "Perfect! TechGiant Corporation is headquartered in Seattle, Washington. So the answer is: Seattle."
                ],
                "answer": "Seattle"
            }
        ])
        
        # 4-hop examples
        examples[4].extend([
            {
                "question": "What university did the CEO of the company that acquired InnovateTech Solutions attend?",
                "documents": [
                    {
                        "title": "InnovateTech Solutions",
                        "text": "InnovateTech Solutions was a software startup founded in 2018 that developed artificial intelligence tools."
                    },
                    {
                        "title": "MegaCorp Industries Acquisitions",
                        "text": "MegaCorp Industries acquired InnovateTech Solutions in 2021 for $75 million."
                    },
                    {
                        "title": "MegaCorp Industries Leadership",
                        "text": "MegaCorp Industries is led by CEO Jennifer Wang, who has been with the company since 2019."
                    },
                    {
                        "title": "Jennifer Wang Executive Profile",
                        "text": "Jennifer Wang earned her MBA from MIT Sloan School of Management and her undergraduate degree from UC Berkeley."
                    }
                ],
                "reasoning_steps": [
                    "I need to find out which company acquired InnovateTech Solutions first.",
                    "Found the acquiring company: MegaCorp Industries. Now I need to identify who the CEO of MegaCorp Industries is.",
                    "Found the CEO: Jennifer Wang. Now I need to find Jennifer Wang's educational background.",
                    "Perfect! I have the complete chain: InnovateTech Solutions → acquired by MegaCorp Industries → CEO Jennifer Wang → attended MIT Sloan School of Management. So the answer is: MIT Sloan School of Management."
                ],
                "answer": "MIT Sloan School of Management"
            }
        ])
        
        # Unanswerable examples
        examples["unanswerable"].extend([
            {
                "question": "What is the birth year of the CEO's spouse of the company that acquired TechCorp Industries?",
                "documents": [
                    {
                        "title": "TechCorp Industries",
                        "text": "TechCorp Industries is a technology company founded in 2010 that specializes in software development."
                    },
                    {
                        "title": "Business Acquisitions",
                        "text": "Business acquisitions involve one company purchasing another company or its assets."
                    }
                ],
                "reasoning_steps": [
                    "I need to find which company acquired TechCorp Industries first.",
                    "I searched for acquisition information about TechCorp Industries, but the available context does not contain any details about which company acquired it. Without knowing the acquiring company, I cannot proceed to find information about its CEO or the CEO's spouse. So the answer is: I cannot find sufficient information to answer this question."
                ],
                "answer": "I cannot find sufficient information to answer this question."
            },
            {
                "question": "What is the current location of the golden sword of Emperor Zarathon?",
                "documents": [
                    {
                        "title": "Ancient Artifacts",
                        "text": "Ancient artifacts are objects made by humans in the past, often studied by archaeologists."
                    },
                    {
                        "title": "Museum Collections",
                        "text": "Museums collect and display various artifacts and specimens for public education."
                    }
                ],
                "reasoning_steps": [
                    "I need to find information about Emperor Zarathon and his golden sword. The provided context discusses ancient artifacts and museums in general, but does not mention Emperor Zarathon or any specific golden sword. So the answer is: I cannot find information about Emperor Zarathon or his golden sword in the available sources."
                ],
                "answer": "I cannot find information about Emperor Zarathon or his golden sword in the available sources."
            }
        ])
        
        return dict(examples)
    
    def get_examples(self, 
                     num_hops: int = 2,
                     num_examples: int = 3,
                     include_unanswerable: bool = True) -> List[Dict[str, Any]]:
        """
        Get examples based on criteria.
        
        Args:
            num_hops: Target number of hops
            num_examples: Number of examples to return
            include_unanswerable: Whether to include unanswerable examples
            
        Returns:
            List of example dictionaries
        """
        selected_examples = []
        
        # Get examples for the target hop count
        if num_hops in self.examples:
            hop_examples = self.examples[num_hops]
            # Take up to num_examples - 1 to leave room for unanswerable
            take_count = num_examples - 1 if include_unanswerable else num_examples
            selected_examples.extend(
                random.sample(hop_examples, min(take_count, len(hop_examples)))
            )
        
        # Add examples from nearby hop counts if needed
        if len(selected_examples) < num_examples - (1 if include_unanswerable else 0):
            # Try adjacent hop counts
            for hop_delta in [1, -1, 2, -2]:
                alt_hops = num_hops + hop_delta
                if alt_hops in self.examples and alt_hops > 0:
                    remaining_needed = num_examples - len(selected_examples) - (1 if include_unanswerable else 0)
                    hop_examples = self.examples[alt_hops]
                    selected_examples.extend(
                        random.sample(hop_examples, min(remaining_needed, len(hop_examples)))
                    )
                    if len(selected_examples) >= num_examples - (1 if include_unanswerable else 0):
                        break
        
        # Add one unanswerable example if requested
        if include_unanswerable and "unanswerable" in self.examples:
            unanswerable_examples = self.examples["unanswerable"]
            if unanswerable_examples:
                selected_examples.append(random.choice(unanswerable_examples))
        
        # Shuffle the final selection
        random.shuffle(selected_examples)
        
        return selected_examples[:num_examples]
    
    def get_example_by_pattern(self, pattern: str) -> Optional[Dict[str, Any]]:
        """
        Get an example matching a specific pattern.
        
        Args:
            pattern: Pattern to match (e.g., "founder", "acquisition", "director")
            
        Returns:
            Matching example or None
        """
        pattern_lower = pattern.lower()
        
        for hop_count, examples in self.examples.items():
            if hop_count == "unanswerable":
                continue
            for example in examples:
                if pattern_lower in example["question"].lower():
                    return example
        
        return None
    
    def add_custom_example(self, example: Dict[str, Any], num_hops: int):
        """
        Add a custom example to the bank.
        
        Args:
            example: Example dictionary with required fields
            num_hops: Number of hops for categorization
        """
        required_fields = ["question", "documents", "reasoning_steps", "answer"]
        if not all(field in example for field in required_fields):
            raise ValueError(f"Example must contain all required fields: {required_fields}")
        
        if num_hops not in self.examples:
            self.examples[num_hops] = []
        
        self.examples[num_hops].append(example) 