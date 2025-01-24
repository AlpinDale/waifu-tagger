#!/usr/bin/env python3
"""
requires-python = ">=3.8"
dependencies = [
    "fastapi>=0.104.1",
    "python-multipart>=0.0.6",
    "uvicorn>=0.24.0",
    "pillow>=10.1.0",
    "numpy>=1.24.0",
    "pandas>=2.1.3",
    "onnxruntime>=1.16.3",
    "huggingface-hub>=0.19.4",
]
optional-dependencies = {
    "gpu" = [
        "onnxruntime-gpu>=1.16.3",
    ]
}
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from PIL import Image


class CustomFormatter(logging.Formatter):
    level_name_colors = {
        logging.DEBUG: '\x1b[38;20m',    # gray
        logging.INFO: '\x1b[32m',        # green
        logging.WARNING: '\x1b[33;20m',  # yellow
        logging.ERROR: '\x1b[31;20m',    # red
        logging.CRITICAL: '\x1b[31;1m',  # bold red
    }
    reset = '\x1b[0m'

    def format(self, record):
        if not hasattr(record, 'levelprefix'):
            color = self.level_name_colors.get(record.levelno, '')
            record.levelprefix = f"{color}{record.levelname}:    {self.reset}"
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter('%(levelprefix)s %(message)s'))
logger.handlers = [handler]

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_tagger(
        model_file=app.state.args.model_file,
        tags_file=app.state.tags_file,
        model_repo=app.state.model_repo,
        backend=app.state.backend
    )
    yield
    pass

app = FastAPI(lifespan=lifespan)
tagger = None

def download_file(repo: str, filename: str, output_path: str):
    path = huggingface_hub.hf_hub_download(
        repo,
        filename,
    )
    if not os.path.exists(output_path):
        os.symlink(path, output_path)
    return path

class Tagger:
    def __init__(self, model_path: str, labels_path: str, backend: str = "cpu"):
        start_time = time.time()

        if backend == "cpu":
            providers = ['CPUExecutionProvider']
        elif backend == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif backend == "tensorrt":
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            logger.warning(f"Unknown backend '{backend}', falling back to CPU")
            providers = ['CPUExecutionProvider']

        available_providers = rt.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        if backend != "cpu" and providers[0] not in available_providers:
            logger.warning(f"{providers[0]} not available. Available providers: {available_providers}")
            if backend == "tensorrt":
                logger.warning("Falling back to CUDA")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if backend == "cuda" or (backend == "tensorrt" and 'CUDAExecutionProvider' not in available_providers):
                logger.warning("Falling back to CPU")
                providers = ['CPUExecutionProvider']

        try:
            logger.info(f"Attempting to load model with providers: {providers}")
            self.model = rt.InferenceSession(model_path, providers=providers)
            active_provider = self.model.get_providers()[0]
            logger.info(f"Model loaded with {active_provider}")

            if backend != "cpu" and active_provider == 'CPUExecutionProvider':
                logger.warning(f"{backend} was requested but model loaded on CPU")
                logger.info(f"Active providers: {self.model.get_providers()}")
        except Exception as e:
            logger.error(f"Provider initialization failed: {str(e)}")
            if backend != "cpu":
                logger.warning("Falling back to CPU")
                self.model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        logger.info(f"Model loading took: {time.time() - start_time:.2f}s")

        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

        logger.info(f"Loading labels from {labels_path}")
        labels_start = time.time()
        tags_df = pd.read_csv(labels_path)
        self.tag_names = tags_df["name"].tolist()
        self.rating_indexes = list(np.where(tags_df["category"] == 9)[0])
        self.general_indexes = list(np.where(tags_df["category"] == 0)[0])
        self.character_indexes = list(np.where(tags_df["category"] == 4)[0])
        logger.info(f"Loaded {len(self.tag_names)} tags")
        logger.info(f"Labels loading took: {time.time() - labels_start:.2f}s")

    def prepare_image(self, image):
        target_size = self.model_target_size

        if image.mode != "RGB":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")

        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.Resampling.BICUBIC,
            )

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]

        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image, general_threshold=0.35, character_threshold=0.85):
        image_array = self.prepare_image(image)

        inference_start = time.time()
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_array})[0]
        logger.info(f"Model inference took: {time.time() - inference_start:.2f}s")

        probs = preds[0].astype(float)

        ratings = [(self.tag_names[i], probs[i]) for i in self.rating_indexes]
        ratings.sort(key=lambda x: x[1], reverse=True)

        general = [(self.tag_names[i], probs[i]) for i in self.general_indexes 
                  if probs[i] > general_threshold]
        general.sort(key=lambda x: x[1], reverse=True)

        characters = [(self.tag_names[i], probs[i]) for i in self.character_indexes 
                     if probs[i] > character_threshold]
        characters.sort(key=lambda x: x[1], reverse=True)

        return general, characters, ratings

def initialize_tagger(model_file="model.onnx",
                     tags_file="selected_tags.csv",
                     model_repo=None,
                     backend="cpu"):
    global tagger

    if not os.path.exists(model_file):
        if not model_repo:
            raise ValueError(f"Model file '{model_file}' not found and no model_repo specified")
        logger.info(f"Downloading model from {model_repo}...")
        download_file(model_repo, model_file, model_file)

    if not os.path.exists(tags_file):
        if not model_repo:
            raise ValueError(f"Tags file '{tags_file}' not found and no model_repo specified")
        logger.info(f"Downloading tags from {model_repo}...")
        download_file(model_repo, "selected_tags.csv", tags_file)

    logger.info("Initializing tagger...")
    tagger = Tagger(model_file, tags_file, backend)

@app.post("/")
async def tag_image(file: UploadFile = File(...)):
    if not tagger:
        return {"error": "Tagger not initialized"}

    start_time = time.time()
    logger.info(f"Processing image: {file.filename}")

    contents = await file.read()
    image = Image.open(BytesIO(contents))

    process_start = time.time()
    general, characters, ratings = tagger.predict(image)
    logger.info(f"Processed {file.filename} in {time.time() - process_start:.2f}s")

    result = {
        "ratings": [{
            "name": name,
            "confidence": float(conf)
        } for name, conf in ratings],
        "general": [{
            "name": name,
            "confidence": float(conf)
        } for name, conf in general],
        "characters": [{
            "name": name,
            "confidence": float(conf)
        } for name, conf in characters],
        "timing": {
            "processing_time": time.time() - process_start,
            "total_time": time.time() - start_time
        }
    }

    return result

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description='Image tagging server')

    parser.add_argument('--host', default='0.0.0.0',
                      help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to bind the server to (default: 8000)')

    parser.add_argument('--log-level', default='INFO',
                      help='Logging level (default: INFO)')

    parser.add_argument('--model-repo', 
                      default=None,
                      help='HuggingFace model repository (optional, used if model files not found locally)')
    parser.add_argument('--model-file', default='model.onnx',
                      help='Model filename (default: model.onnx)')
    parser.add_argument('--tags-file', default='selected_tags.csv',
                      help='Tags filename (default: selected_tags.csv)')

    parser.add_argument('--backend', 
                      choices=['cpu', 'cuda', 'tensorrt'],
                      default='cpu',
                      help='Backend to use for inference (default: cpu)')

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    app.state.args = args
    app.state.model_repo = args.model_repo
    app.state.tags_file = args.tags_file
    app.state.backend = args.backend

    uvicorn.run(app, host=args.host, port=args.port)
