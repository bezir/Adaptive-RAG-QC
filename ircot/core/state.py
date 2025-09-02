#!/usr/bin/env python3
"""
IRCOT State Management - Tracks the state throughout the IRCOT process.

This module manages:
- Retrieved documents and their deduplication
- Reasoning chain history
- Retrieval queries and results
- Current iteration state
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """Represents a retrieved document."""
    title: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.title, self.text))
    
    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.title == other.title and self.text == other.text


@dataclass
class RetrievalRecord:
    """Records a single retrieval operation."""
    query: str
    documents: List[Document]
    stage: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IRCoTState:
    """
    Manages the complete state of an IRCOT execution.
    
    This class tracks:
    - The original question
    - All retrieved documents (with deduplication)
    - The reasoning chain being built
    - Retrieval history
    - Configuration parameters
    - Current iteration count
    """
    
    def __init__(self, question: str, config: Dict[str, Any]):
        """
        Initialize IRCOT state.
        
        Args:
            question: The original question being answered
            config: Configuration parameters for this run
        """
        self.question = question
        self.config = config
        
        # Document management
        self._documents: Dict[str, Document] = {}  # title -> Document
        self._document_order: List[str] = []  # Ordered list of titles
        
        # Reasoning chain
        self.reasoning_chain: List[str] = []
        
        # Retrieval history
        self.retrieval_history: List[RetrievalRecord] = []
        
        # Iteration tracking
        self.iteration_count = 0
        
        # Answer tracking
        self.final_answer: Optional[str] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
    
    def add_documents(self, documents: List[Dict[str, Any]], query: str, stage: str):
        """
        Add retrieved documents to the state with deduplication and sliding window management.
        
        Args:
            documents: List of document dictionaries with 'title' and 'text' keys
            query: The query used for retrieval
            stage: Stage identifier (e.g., 'initial', 'iteration_1')
        """
        # DEBUG: Log document addition
        import threading
        worker_id = f"worker_{threading.current_thread().ident}"
        state_instance_id = id(self)
        
        # Simple debug logging (no hardcoded contamination detection)
        # Could add logging here if needed
        doc_objects = []
        new_docs_added = 0
        
        for doc in documents:
            # Create completely isolated Document object with deep copy
            import copy
            doc_obj = Document(
                title=copy.deepcopy(doc.get("title", "")),
                text=copy.deepcopy(doc.get("text", "")),
                score=doc.get("score", 1.0),
                metadata=copy.deepcopy(doc.get("metadata", {}))
            )
            
            # Add to state if not duplicate
            if doc_obj.title not in self._documents:
                self._documents[doc_obj.title] = doc_obj
                self._document_order.append(doc_obj.title)
                new_docs_added += 1
            
            doc_objects.append(doc_obj)
        
        # Implement aggressive sliding window: maintain strict document limit
        max_docs = self.config.get("max_total_docs", 25)  # Reduced default limit
        
        # Always maintain the limit - remove excess documents
        while len(self._document_order) > max_docs:
            # Remove oldest document
            if self._document_order:
                oldest_title = self._document_order.pop(0)
                if oldest_title in self._documents:
                    del self._documents[oldest_title]
        
        # Log sliding window activity
        if new_docs_added > 0 and len(self._document_order) == max_docs:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"📚 Sliding window active: {len(self._document_order)}/{max_docs} documents maintained")
        
        # Record retrieval
        self.retrieval_history.append(
            RetrievalRecord(
                query=query,
                documents=doc_objects,
                stage=stage
            )
        )
    
    def add_reasoning_step(self, step: str):
        """Add a reasoning step to the chain."""
        self.reasoning_chain.append(step.strip())
    
    def get_all_documents(self) -> List[Document]:
        """Get all unique documents in order of retrieval."""
        return [self._documents[title] for title in self._document_order]
    
    def get_retrieved_titles(self) -> Set[str]:
        """Get set of all retrieved document titles."""
        return set(self._documents.keys())
    
    def get_context_for_prompt(self, max_docs: Optional[int] = None, max_chars_per_doc: int = 3000) -> str:
        """
        Format documents as context for prompting.
        
        Args:
            max_docs: Maximum number of documents to include
            max_chars_per_doc: Maximum characters per document (default: 2000)
            
        Returns:
            Formatted context string
        """
        docs = self.get_all_documents()
        
        if max_docs:
            docs = docs[:max_docs]
        
        context_parts = []
        total_chars = 0
        max_total_chars = max_chars_per_doc * (max_docs or len(docs)) * 1.2  # Allow 20% overhead
        
        for doc in docs:
            title_line = f"Wikipedia Title: {doc.title}"
            context_parts.append(title_line)
            total_chars += len(title_line)
            
            # Intelligent document truncation - prioritize beginning and end
            doc_text = doc.text.strip()
            remaining_budget = max_total_chars - total_chars
            
            if len(doc_text) <= max_chars_per_doc and total_chars + len(doc_text) < max_total_chars:
                # Full document fits
                context_parts.append(doc_text)
                total_chars += len(doc_text)
            else:
                # Smart truncation: keep beginning + end, indicate middle truncation
                available_chars = min(max_chars_per_doc, remaining_budget - 50)  # Reserve 50 for truncation message
                if available_chars > 200:  # Only truncate if we have meaningful space
                    start_chars = int(available_chars * 0.7)  # 70% from beginning
                    end_chars = available_chars - start_chars - 20  # 30% from end, reserve for message
                    
                    truncated_text = (
                        doc_text[:start_chars] + 
                        " ... [middle truncated] ... " + 
                        doc_text[-end_chars:] if end_chars > 0 else ""
                    )
                    context_parts.append(truncated_text)
                    total_chars += len(truncated_text)
                else:
                    # Very limited space - just beginning
                    truncated_text = doc_text[:available_chars] + "... [truncated]" if available_chars > 0 else "[truncated]"
                    context_parts.append(truncated_text)
                    total_chars += len(truncated_text)
            
            context_parts.append("")  # Empty line between documents
            total_chars += 1
            
            # Stop if we're approaching limits
            if total_chars > max_total_chars * 0.9:
                break
        
        return "\n".join(context_parts).strip()
    
    def get_full_reasoning_chain(self) -> str:
        """Get the complete reasoning chain as a single string."""
        return " ".join(self.reasoning_chain)
    
    def get_last_reasoning_steps(self, n: int = 3) -> List[str]:
        """Get the last n reasoning steps."""
        return self.reasoning_chain[-n:] if len(self.reasoning_chain) >= n else self.reasoning_chain
    
    def set_answer(self, answer: str):
        """Set the final answer."""
        self.final_answer = answer.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "question": self.question,
            "config": self.config,
            "documents": [
                {
                    "title": doc.title,
                    "text": doc.text,
                    "score": doc.score,
                    "metadata": doc.metadata
                }
                for doc in self.get_all_documents()
            ],
            "reasoning_chain": self.reasoning_chain,
            "retrieval_history": [
                {
                    "query": record.query,
                    "stage": record.stage,
                    "timestamp": record.timestamp.isoformat(),
                    "num_docs": len(record.documents)
                }
                for record in self.retrieval_history
            ],
            "iteration_count": self.iteration_count,
            "final_answer": self.final_answer,
            "metadata": self.metadata
        }
    
    def __repr__(self):
        return (
            f"IRCoTState(question='{self.question[:50]}...', "
            f"iterations={self.iteration_count}, "
            f"docs={len(self._documents)}, "
            f"reasoning_steps={len(self.reasoning_chain)})"
        ) 