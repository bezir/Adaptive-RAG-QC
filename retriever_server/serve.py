from time import perf_counter
from fastapi import FastAPI, Request
import threading
import logging

from unified_retriever import UnifiedRetriever

app = FastAPI()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server with worker isolation."}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "retriever_server"}


@app.post("/retrieve")
@app.post("/retrieve/")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    
    # Extract worker identification if provided
    worker_id = arguments.pop("worker_id", None)
    thread_id = threading.current_thread().ident
    request_id = f"worker_{worker_id}_thread_{thread_id}" if worker_id else f"thread_{thread_id}"
    
    logger.info(f"🔍 RETRIEVAL DEBUG: {request_id} processing request")
    
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    
    # CRITICAL FIX: Create per-request retriever instance to prevent contamination
    retriever = UnifiedRetriever(host="http://localhost/", port=9200)
    logger.debug(f"🔍 RETRIEVAL DEBUG: {request_id} created isolated retriever instance {id(retriever)}")
    
    start_time = perf_counter()
    
    retrieval = getattr(retriever, retrieval_method)(**arguments)
    
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    
    logger.info(f"✅ RETRIEVAL DEBUG: {request_id} completed in {time_in_seconds}s, returned {len(retrieval) if retrieval else 0} docs")
    
    return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}
