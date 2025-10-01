from pathlib import Path
from typing import Any, Dict, List, Optional

import huggingface_hub
import numpy as np
import pandas as pd
import timm
import torch
from timm.data.config import resolve_data_config
from settings import Settings
from shared import logger

# --- Global variable to hold the tagger instance in each worker process ---
worker_tagger: Optional["Tagger"] = None


# --- Core Tagger Class (No changes needed) ---
class Tagger:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(settings.device)
        self.model_dir = Path("models") / self.settings.model_repo.replace("/", "_")

        self.tag_names: np.ndarray
        self.per_tag_thresholds: torch.Tensor | None = None
        self.categories: Dict[int, Dict[str, Any]] = {}

        self._download_files()
        self._load_tags_and_thresholds()
        self._load_model()

    def _download_files(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checking for model files in {self.model_dir}...")
        for filename in [
            self.settings.tags_filename,
            self.settings.thresholds_filename,
        ]:
            try:
                huggingface_hub.hf_hub_download(
                    self.settings.model_repo,
                    filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    token=self.settings.token if self.settings.token else None,
                )
            except huggingface_hub.utils.HfHubHTTPError:
                logger.warning(
                    f"Optional file '{filename}' not found in repo, skipping."
                )

    def recall_first_thr(self, row):
        p, r = row["test_precision"], row["test_recall"]
        if pd.isna(p) or pd.isna(r):
            # tiny-count tags – clamp low
            return 0.15
        # binary search F1 curve (simplified; you can pre-compute if needed)
        # we pick threshold where recall ≥ recall_target if possible
        thr = row["best_threshold"]
        # crude heuristic: move threshold lower until recall hits target or precision bottoms out
        while r < 0.90 and p > 0.5 and thr > 0.01:
            thr = max(thr - 0.02, 0.01)
            r += 0.02  # lazy linear approx; replace by true PR data if available
        return thr

    def _load_tags_and_thresholds(self):
        """
        Loads tags, category data, and both category-level and per-tag thresholds.
        """
        tags_path = self.model_dir / self.settings.tags_filename
        tags_df = pd.read_csv(tags_path)

        self.tag_names = tags_df["name"].to_numpy()
        logger.info(f"Loaded {len(self.tag_names)} total tags.")

        if "best_threshold" in tags_df.columns:
            if self.settings.use_recall:
                logger.info(
                    "Found 'best_threshold' column & 'use_recall' is prefered, loading per-tag computed recall."
                )
                tags_df["recall_threshold"] = tags_df.apply(
                    self.recall_first_thr, axis=1
                )

                self.per_tag_thresholds = (
                    torch.from_numpy(
                        tags_df["recall_threshold"].to_numpy(dtype=np.float32)
                    )
                    .to(self.device)
                    .view(1, -1)
                )
            else:
                logger.info(
                    "Found 'best_threshold' column, loading per-tag thresholds."
                )
                self.per_tag_thresholds = (
                    torch.from_numpy(
                        tags_df["best_threshold"].to_numpy(dtype=np.float32)
                    )
                    .to(self.device)
                    .view(1, -1)
                )

        default_categories = {
            0: {"name": "general", "threshold": 0.35},
            4: {"name": "character", "threshold": 0.75},
            9: {"name": "rating", "threshold": 0},
        }
        thresholds_path = self.model_dir / self.settings.thresholds_filename
        if thresholds_path.exists():
            logger.info(f"Loading category thresholds from '{thresholds_path}'.")
            for item in pd.read_csv(thresholds_path).to_dict("records"):
                self.categories[item["category"]] = {
                    "name": item["name"],
                    "threshold": item["threshold"],
                }
        else:
            logger.warning("No category thresholds file found. Using defaults.")
            self.categories = default_categories

        for cat_id in self.categories:
            indices = np.where(tags_df["category"] == cat_id)[0]
            self.categories[cat_id]["indices"] = torch.from_numpy(indices).to(
                self.device
            )
        logger.info(f"Loaded {len(self.categories)} tag categories.")

    def _load_model(self):
        logger.info(
            f"Loading model '{self.settings.model_repo}' on device '{self.device}'..."
        )
        hf_model_name = f"hf-hub:{self.settings.model_repo}"
        self.model = (
            timm.create_model(
                hf_model_name,
                pretrained=True,
            )
            .to(self.device)
            .eval()
        )
        if self.settings.compile_model and self.device.type == "cuda":
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
        data_config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.mean = torch.tensor(data_config["mean"], device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor(data_config["std"], device=self.device).view(1, 3, 1, 1)
        self.permute_channels = (
            True
            if self.settings.model_repo.lower().startswith("smilingwolf/")
            else False
        )
        logger.info(f"Mean: {self.mean} | Std: {self.std}")
        logger.info("Model loaded and ready.")

    def _preprocess_batch(self, batch_nhwc: np.ndarray) -> torch.Tensor:
        if batch_nhwc.ndim != 4:
            raise ValueError(
                f"Expected a 4D array (N, H, W, C), but got {batch_nhwc.ndim}D."
            )
        batch_tensor = torch.from_numpy(batch_nhwc).to(self.device).permute(0, 3, 1, 2)
        batch_tensor = (batch_tensor.float() / 255.0 - self.mean) / self.std
        return batch_tensor

    def predict(
        self, batch_nhwc: np.ndarray, use_best_thresholds: bool
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        if not isinstance(batch_nhwc, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        batch_tensor = self._preprocess_batch(batch_nhwc)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type):
            probs = torch.sigmoid(self.model(batch_tensor))
        return self._format_output(probs, use_best_thresholds)

    def _format_output(
        self, probs: torch.Tensor, use_best_thresholds: bool
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        batch_size = probs.shape[0]
        results = {
            cat_info["name"]: [[] for _ in range(batch_size)]
            for cat_info in self.categories.values()
        }
        all_tags = []

        for cat_info in self.categories.values():
            cat_indices = cat_info["indices"]
            cat_probs = probs[:, cat_indices]
            mask = cat_probs > cat_info["threshold"]
            if use_best_thresholds and self.per_tag_thresholds is not None:
                per_tag_thresh = self.per_tag_thresholds[:, cat_indices]
                mask &= cat_probs > per_tag_thresh
            batch_indices, local_tag_indices = torch.where(mask)
            if batch_indices.shape[0] > 0:
                global_tag_indices = cat_indices[local_tag_indices]
                scores = cat_probs[batch_indices, local_tag_indices]
                all_tags.append(
                    {
                        "category_name": cat_info["name"],
                        "batch_indices": batch_indices,
                        "tag_indices": global_tag_indices,
                        "scores": scores,
                    }
                )

        if not all_tags:
            return results

        combined_batch_indices = (
            torch.cat([t["batch_indices"] for t in all_tags]).cpu().numpy()
        )
        combined_tag_indices = (
            torch.cat([t["tag_indices"] for t in all_tags]).cpu().numpy()
        )
        combined_scores = torch.cat([t["scores"] for t in all_tags]).cpu().numpy()
        combined_cat_names = np.concatenate(
            [np.full(len(t["scores"]), t["category_name"]) for t in all_tags]
        )

        lex_sort_keys = (-combined_scores, combined_batch_indices)
        sorted_indices = np.lexsort(lex_sort_keys)

        for i in sorted_indices:
            batch_idx = combined_batch_indices[i]
            cat_name = combined_cat_names[i]
            tag_idx = combined_tag_indices[i]
            score = combined_scores[i]
            results[cat_name][batch_idx].append(
                {"name": self.tag_names[tag_idx], "confidence": float(score)}
            )

        return results
