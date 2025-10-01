import asyncio
import multiprocessing
import multiprocessing.pool
import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from settings import Settings
from tagger import Tagger

def initialize_worker(
    settings_dict: dict[str, Any],
    devices: list[str],
    worker_id_counter,
):
    """
    Initializer for each worker process in the pool.
    Atomically gets a unique ID to select its device from the list.
    """
    global worker_tagger, logger

    # --- NEW: Robustly acquire a unique worker ID ---
    with worker_id_counter.get_lock():
        worker_id = worker_id_counter.value
        worker_id_counter.value += 1
    
    # Use the acquired ID to get the device for this specific worker
    # The modulo operator adds robustness in case workers are ever restarted
    device_id = devices[worker_id % len(devices)]

    # Each worker needs its own logger setup.
    logger = setup_logging(settings_dict.get("log_level", "INFO"))

    logger.info(f"Initializing worker #{worker_id} on device: {device_id}")
    try:
        worker_settings = Settings.model_construct(**settings_dict)
        if worker_settings.token:
            os.environ["HF_TOKEN"] = str(worker_settings.token)
        
        # Override the device setting for this specific worker
        worker_settings.device = device_id
        
        worker_tagger = Tagger(worker_settings)
        logger.info(f"Worker #{worker_id} on {device_id} initialized successfully.")
    except Exception as e:
        logger.critical(
            f"FATAL: Worker #{worker_id} initialization on {device_id} failed: {e}",
            exc_info=True,
        )
        worker_tagger = None



# --- API & Main Execution (Updated for Multiprocessing) ---
class TagPrediction(BaseModel):
    name: str
    confidence: float


class CategorizedTaggingResponse(BaseModel):
    results: List[Dict[str, List[TagPrediction]]]


def run_prediction_on_worker(
    contents: bytes, use_best_thresholds: bool
) -> tuple[CategorizedTaggingResponse | None, str|None]:
    """
    The target function executed by the worker pool for each prediction task.
    """
    global worker_tagger
    if worker_tagger is None:
        raise RuntimeError("Tagger worker is not initialized.")
    
    
    try:
        with np.load(BytesIO(contents)) as npz:
            image_batch = npz[npz.files[0]]
    except Exception as e:
        return None, f"Could not read .npz file: {e}"
    
    categorized_results = worker_tagger.predict(image_batch, use_best_thresholds)
    
    batch_size = image_batch.shape[0]
    final_response = []
    for i in range(batch_size):
        item_result = {
            cat: categorized_results[cat][i] for cat in categorized_results
        }
        final_response.append(item_result)

    return CategorizedTaggingResponse(results=final_response), None

def create_app(settings: Settings) -> FastAPI:
    # --- Lifespan manager for the worker pool ---
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings: Settings = app.state.settings

        # Use 'spawn' context for CUDA safety
        spawn_ctx = multiprocessing.get_context("spawn")

        # --- Logic for determining workers and devices (from previous answer) ---
        if settings.device_ids is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus > 0:
                logger.info(f"Auto-detecting GPUs: Found {num_gpus} available.")
                devices = [f"cuda:{i}" for i in range(num_gpus)]
            else:
                logger.info("Auto-detecting GPUs: None found. Falling back to CPU.")
                devices = ["cpu"]
        else:
            if settings.device_ids:
                logger.info(f"Using specified GPU device IDs: {settings.device_ids}")
                devices = [f"cuda:{i}" for i in settings.device_ids]
            else:
                logger.info("Empty device-ids list provided. Using CPU.")
                devices = ["cpu"]
        
        num_workers = len(devices)
        if num_workers == 1 and devices[0] == "cpu":
            logger.info("Configuration results in 1 CPU worker.")
        else:
            logger.info(f"Spawning {num_workers} worker(s) for devices: {devices}")
        
        # --- NEW: Create a shared, thread-safe counter for worker IDs ---
        worker_id_counter = spawn_ctx.Value('i', 0)

        # Pass settings dict, devices list, and the new counter to each worker
        init_args = [settings.model_dump(), devices, worker_id_counter]

        try:
            app.state.pool = spawn_ctx.Pool(
                processes=num_workers,
                initializer=initialize_worker,
                initargs=init_args,
            )
            logger.info("Worker pool started successfully.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to create worker pool: {e}", exc_info=True)
            app.state.pool = None
            sys.exit(1)

        yield  # Application runs here

        logger.info("Shutting down worker pool...")
        if app.state.pool:
            app.state.pool.close()
            app.state.pool.join()
        logger.info("Worker pool shut down.")

    app = FastAPI(
        title="High-Throughput Tagger API",
        version="6.0.0-multi-gpu",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.post(
        "/tag",
        response_model=CategorizedTaggingResponse,
        response_class=ORJSONResponse,
        summary="Tag a batch of images from a NumPy array",
    )
    async def tag_batch(
        file: UploadFile = File(
            ..., description="A .npz file containing one NHWC RGB NumPy array."
        ),
        use_best_thresholds: bool = Query(
            False,
            description="Use pre-defined, per-tag 'best_thresholds' in addition to category thresholds.",
        ),
    ):
        pool: multiprocessing.pool.Pool = app.state.pool
        if pool is None:
            raise HTTPException(
                status_code=503, detail="Tagger worker pool is not available."
            )

        if not file.filename.endswith(".npz"):
            raise HTTPException(
                status_code=400, detail="Only .npz files are supported."
            )

        loop = asyncio.get_running_loop()

        # Apply the task to the worker pool asynchronously
        async_result = pool.apply_async(
            run_prediction_on_worker, (await file.read(), use_best_thresholds)
        )
        categorized_results, error = await loop.run_in_executor(None, async_result.get)
        
        if error or not categorized_results:
            raise HTTPException(400, str(error))

        return categorized_results
        

    @app.get("/status", summary="Get server status", response_class=ORJSONResponse)
    async def status():
        settings = app.state.settings
        pool = app.state.pool

        if settings.device_ids is None:
            num_configured = (
                torch.cuda.device_count() if torch.cuda.is_available() else 1
            )
            worker_type = "GPU (all available)" if torch.cuda.is_available() else "CPU"
        else:
            num_configured = len(settings.device_ids) if settings.device_ids else 1
            worker_type = "GPU (specific IDs)" if settings.device_ids else "CPU"

        status_info = {
            "status": "ok" if pool else "error",
            "model_repo": settings.model_repo,
            "workers_configured": num_configured,
            "worker_type": worker_type,
            "device_ids_in_use": settings.device_ids,
        }
        return status_info

    return app


settings = Settings()
app = create_app(settings)

if __name__ == "__main__":
    # NOTE: When running this script directly, Uvicorn will start a SINGLE main process.
    # This main process will then spawn the worker processes using the 'multiprocessing.Pool'.
    # You should NOT use Uvicorn's --workers flag with this setup, as it would create
    # multiple independent main processes, each with its own worker pool, which is redundant
    # and memory-intensive.

    # Force 'spawn' start method if not on Windows, for CUDA safety.
    if sys.platform != "win32":
        multiprocessing.set_start_method("spawn", force=True)

    print("--- Starting Tagger Server with Multiprocessing Worker Pool ---")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )
