"""
Prefetching mechanism for parallel data loading and processing
Optimizes RAG systems by prefetching retrieval results
"""
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Any
from threading import Thread
from queue import Queue
import time


class PrefetchDataLoader:
    """
    DataLoader with prefetching for parallel data loading.
    Reduces GPU idle time by prefetching batches in background threads.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        prefetch_factor: int = 2,
        device: torch.device = None,
    ):
        """
        Args:
            dataloader: Base DataLoader to wrap
            prefetch_factor: Number of batches to prefetch
            device: Device to prefetch batches to
        """
        self.dataloader = dataloader
        self.prefetch_factor = prefetch_factor
        self.device = device
        
        self.queue = Queue(maxsize=prefetch_factor)
        self.thread = None
        self._stop_thread = False
    
    def _prefetch_worker(self):
        """Worker thread that prefetches batches."""
        for batch in self.dataloader:
            if self._stop_thread:
                break
            
            # Move to device if specified
            if self.device is not None:
                batch = {k: v.to(self.device, non_blocking=True) 
                        for k, v in batch.items()}
            
            self.queue.put(batch)
        
        self.queue.put(None)  # Signal end of data
    
    def __iter__(self):
        """Start prefetching thread and return iterator."""
        self._stop_thread = False
        self.thread = Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
        return self
    
    def __next__(self):
        """Get next prefetched batch."""
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch
    
    def __len__(self):
        """Return length of underlying dataloader."""
        return len(self.dataloader)
    
    def stop(self):
        """Stop prefetching thread."""
        self._stop_thread = True
        if self.thread is not None:
            self.thread.join()


class LookaheadRetriever:
    """
    Lookahead retrieval mechanism for RAG systems.
    Prefetches retrieval results for anticipated queries.
    """
    
    def __init__(
        self,
        retrieval_fn: Callable[[str], List[Dict]],
        lookahead_window: int = 3,
        prefetch_queue_size: int = 10,
    ):
        """
        Args:
            retrieval_fn: Function that takes a query and returns retrieved documents
            lookahead_window: Number of queries to look ahead
            prefetch_queue_size: Maximum size of prefetch queue
        """
        self.retrieval_fn = retrieval_fn
        self.lookahead_window = lookahead_window
        self.prefetch_queue_size = prefetch_queue_size
        
        self.prefetch_queue: Queue = Queue(maxsize=prefetch_queue_size)
        self.prefetch_thread: Optional[Thread] = None
        self._stop_thread = False
    
    def _prefetch_worker(self, query_queue: Queue):
        """Worker thread that prefetches retrieval results."""
        while not self._stop_thread:
            try:
                query = query_queue.get(timeout=1.0)
                if query is None:
                    break
                
                # Perform retrieval
                results = self.retrieval_fn(query)
                
                # Add to prefetch queue
                try:
                    self.prefetch_queue.put((query, results), timeout=0.1)
                except:
                    pass  # Queue full, skip
                    
            except:
                continue
    
    def start_prefetching(self, query_stream: List[str]):
        """Start prefetching retrieval results for query stream."""
        query_queue = Queue()
        
        # Add queries to queue
        for query in query_stream:
            query_queue.put(query)
        query_queue.put(None)  # Signal end
        
        self._stop_thread = False
        self.prefetch_thread = Thread(target=self._prefetch_worker, args=(query_queue,), daemon=True)
        self.prefetch_thread.start()
    
    def get(self, query: str, timeout: float = 1.0) -> Optional[List[Dict]]:
        """
        Get retrieval results, checking prefetch queue first.
        
        Args:
            query: Query string
            timeout: Timeout for checking prefetch queue
            
        Returns:
            Retrieved documents or None if not found
        """
        # Check prefetch queue
        while not self.prefetch_queue.empty():
            try:
                cached_query, results = self.prefetch_queue.get(timeout=timeout)
                if cached_query == query:
                    return results
                # Put back if not matching
                self.prefetch_queue.put((cached_query, results))
            except:
                break
        
        # Fallback to direct retrieval
        return self.retrieval_fn(query)
    
    def stop(self):
        """Stop prefetching thread."""
        self._stop_thread = True
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()


class BatchPrefetcher:
    """
    Batched prefetching for multiple queries.
    Groups queries into batches for efficient retrieval.
    """
    
    def __init__(
        self,
        batch_retrieval_fn: Callable[[List[str]], List[List[Dict]]],
        batch_size: int = 8,
        prefetch_factor: int = 2,
    ):
        """
        Args:
            batch_retrieval_fn: Function that takes list of queries and returns list of results
            batch_size: Size of batches for retrieval
            prefetch_factor: Number of batches to prefetch
        """
        self.batch_retrieval_fn = batch_retrieval_fn
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        
        self.prefetch_queue: Queue = Queue(maxsize=prefetch_factor)
        self.prefetch_thread: Optional[Thread] = None
        self._stop_thread = False
    
    def _prefetch_worker(self, query_queue: Queue):
        """Worker thread that prefetches batches of retrieval results."""
        batch = []
        
        while not self._stop_thread:
            try:
                query = query_queue.get(timeout=1.0)
                if query is None:
                    # Process remaining batch
                    if batch:
                        results = self.batch_retrieval_fn(batch)
                        for q, r in zip(batch, results):
                            self.prefetch_queue.put((q, r))
                    break
                
                batch.append(query)
                
                # Process batch when full
                if len(batch) >= self.batch_size:
                    results = self.batch_retrieval_fn(batch)
                    for q, r in zip(batch, results):
                        try:
                            self.prefetch_queue.put((q, r), timeout=0.1)
                        except:
                            pass  # Queue full
                    batch = []
                    
            except:
                continue
    
    def start_prefetching(self, query_stream: List[str]):
        """Start prefetching retrieval results for query stream."""
        query_queue = Queue()
        
        for query in query_stream:
            query_queue.put(query)
        query_queue.put(None)  # Signal end
        
        self._stop_thread = False
        self.prefetch_thread = Thread(target=self._prefetch_worker, args=(query_queue,), daemon=True)
        self.prefetch_thread.start()
    
    def get(self, query: str, timeout: float = 1.0) -> Optional[List[Dict]]:
        """
        Get retrieval results from prefetch queue.
        
        Args:
            query: Query string
            timeout: Timeout for checking prefetch queue
            
        Returns:
            Retrieved documents or None if not found
        """
        # Check prefetch queue
        while not self.prefetch_queue.empty():
            try:
                cached_query, results = self.prefetch_queue.get(timeout=timeout)
                if cached_query == query:
                    return results
                # Put back if not matching
                self.prefetch_queue.put((cached_query, results))
            except:
                break
        
        return None
    
    def stop(self):
        """Stop prefetching thread."""
        self._stop_thread = True
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()

