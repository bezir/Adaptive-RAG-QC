#!/usr/bin/env python3
"""
Elasticsearch Retriever - Handles document retrieval for IRCOT.

This module provides:
- BM25 retrieval from Elasticsearch
- Document deduplication
- Query optimization
- Error handling
"""

import os
import logging
import requests
from typing import Dict, Any, List, Optional, Set
import json
from urllib.parse import urljoin


class ElasticsearchRetriever:
    """
    Retrieves documents from Elasticsearch for IRCOT.
    
    Features:
    - BM25 search
    - Corpus selection based on dataset
    - Document formatting
    - Deduplication
    """
    
    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        """
        Initialize the Elasticsearch retriever.
        
        Args:
            host: Elasticsearch host (default from env)
            port: Elasticsearch port (default from env)
            logger: Logger instance
        """
        self.host = host or os.getenv("RETRIEVER_HOST", "http://localhost")
        self.port = port or int(os.getenv("RETRIEVER_PORT", "8000"))
        self.logger = logger or logging.getLogger(__name__)
        
        # Build base URL
        self.base_url = f"{self.host}:{self.port}"
        
        # Corpus mapping
        self.corpus_mapping = {
            'hotpotqa': 'hotpotqa',
            '2wikimultihopqa': '2wikimultihopqa',
            'musique': 'musique',
            'squad': 'wiki',
            'nq': 'wiki',
            'trivia': 'wiki',
            'iirc': 'iirc'
        }
        
        self.logger.info(f"Initialized Elasticsearch retriever: {self.base_url}")
    
    def retrieve(self, 
                 query: str,
                 k: int = 5,
                 dataset: Optional[str] = None,
                 exclude_titles: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 search.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            dataset: Dataset name for corpus selection
            exclude_titles: Set of titles to exclude from results
            
        Returns:
            List of document dictionaries with 'title' and 'text' keys
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to retriever")
            return []
        
        # Determine corpus
        corpus = self._get_corpus_name(dataset)
        
        try:
            # Generate worker identification for request isolation
            import threading
            worker_id = f"worker_{threading.current_thread().ident}"
            
            # Build retrieval request  
            retrieval_params = {
                "query_text": query.strip(),  # Parameter name expected by UnifiedRetriever
                "corpus_name": corpus,  # Parameter name expected by UnifiedRetriever  
                "max_hits_count": k * 2 if exclude_titles else k,  # Get extra if filtering
                "retrieval_method": "retrieve_from_elasticsearch",  # Required by retriever server
                "worker_id": worker_id 
            }
            
            # Debug logging
            self.logger.debug(f"🔍 RETRIEVAL REQUEST: {worker_id} querying '{query[:50]}...' from corpus '{corpus}'")
            
            # Make request to retriever API
            response = self._make_retrieval_request(retrieval_params)
            
            # Parse and format results
            documents = self._parse_retrieval_response(response)
            
            # Filter excluded titles
            if exclude_titles:
                documents = [doc for doc in documents if doc["title"] not in exclude_titles]
            
            # Limit to requested k
            documents = documents[:k]
            
            self.logger.info(f"✅ RETRIEVAL: {worker_id} retrieved {len(documents)} documents for query: '{query[:50]}...'")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []
    
    def _get_corpus_name(self, dataset: Optional[str]) -> str:
        """Get the corpus name for the dataset."""
        if not dataset:
            return "wiki"  # Default corpus
        
        dataset_lower = dataset.lower()
        return self.corpus_mapping.get(dataset_lower, dataset_lower)
    
    def _make_retrieval_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to retriever API."""
        url = urljoin(self.base_url, "/retrieve")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = requests.post(
                url,
                json=params,
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Could not connect to retriever at {url}")
            raise
        except requests.exceptions.Timeout:
            self.logger.error("Retriever request timed out")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"Retriever HTTP error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected retriever error: {e}")
            raise
    
    def _parse_retrieval_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse the retriever API response into document format."""
        documents = []
        
        # Handle different response formats
        if "retrieval" in response:
            results = response["retrieval"]  # Actual key used by retriever API
        elif "results" in response:
            results = response["results"]
        elif "documents" in response:
            results = response["documents"]
        else:
            # Assume response itself is the results list
            results = response if isinstance(response, list) else []
        
        for result in results:
            # Extract document fields
            doc = {
                "title": result.get("title", ""),
                "text": result.get("paragraph_text", result.get("text", result.get("content", ""))),
                "score": result.get("score", 0.0),
                "metadata": {}
            }
            
            # Add any additional metadata
            for key in ["id", "doc_id", "paragraph_id"]:
                if key in result:
                    doc["metadata"][key] = result[key]
            
            # Clean up text
            doc["text"] = self._clean_text(doc["text"])
            
            documents.append(doc)
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean up document text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common artifacts
        text = text.replace("\\n", " ")
        text = text.replace("\\t", " ")
        
        # Truncate very long texts
        max_length = 1000  # characters
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()
    
    def health_check(self) -> bool:
        """Check if the retriever is available."""
        try:
            url = self.base_url
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False 