import os
import time
from PIL import Image
import numpy as np
import onnxruntime as rt
import pandas as pd
import huggingface_hub
import argparse

def download_file(repo: str, filename: str, output_path: str):
    print(f"Downloading from {repo}")
    path = huggingface_hub.hf_hub_download(
        repo,
        filename,
    )
    if not os.path.exists(output_path):
        os.symlink(path, output_path)
    return path

class Tagger:
    def __init__(self, model_path: str, labels_path: str):
        start_time = time.time()
        self.model = rt.InferenceSession(model_path)
        print(f"Model loading took: {time.time() - start_time:.2f}s")

        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height
        print(f"Target size: {self.model_target_size}")

        print(f"Loading labels from {labels_path}")
        labels_start = time.time()
        tags_df = pd.read_csv(labels_path)
        self.tag_names = tags_df["name"].tolist()
        self.rating_indexes = list(np.where(tags_df["category"] == 9)[0])
        self.general_indexes = list(np.where(tags_df["category"] == 0)[0])
        self.character_indexes = list(np.where(tags_df["category"] == 4)[0])
        print(f"Loaded {len(self.tag_names)} tags")
        print(f"Labels loading took: {time.time() - labels_start:.2f}s")

    def prepare_image(self, image):
        start_time = time.time()
        target_size = self.model_target_size

        # Convert to RGB if needed
        if image.mode != "RGB":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")

        # Pad image to square
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]
        
        image_array = np.expand_dims(image_array, axis=0)
        print(f"Image preparation took: {time.time() - start_time:.2f}s")
        return image_array

    def predict(self, image, general_threshold=0.35, character_threshold=0.85):
        prep_start = time.time()
        image_array = self.prepare_image(image)
        print(f"Image preparation took: {time.time() - prep_start:.2f}s")

        inference_start = time.time()
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_array})[0]
        print(f"Model inference took: {time.time() - inference_start:.2f}s")

        postprocess_start = time.time()
        probs = preds[0].astype(float)

        # Process ratings
        ratings = [(self.tag_names[i], probs[i]) for i in self.rating_indexes]
        ratings.sort(key=lambda x: x[1], reverse=True)

        # Process general tags
        general = [(self.tag_names[i], probs[i]) for i in self.general_indexes 
                  if probs[i] > general_threshold]
        general.sort(key=lambda x: x[1], reverse=True)

        # Process character tags
        characters = [(self.tag_names[i], probs[i]) for i in self.character_indexes 
                     if probs[i] > character_threshold]
        characters.sort(key=lambda x: x[1], reverse=True)

        print(f"Post-processing took: {time.time() - postprocess_start:.2f}s")
        return general, characters, ratings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to image file")
    parser.add_argument("-r", "--model-repo", default="SmilingWolf/wd-swinv2-tagger-v3",
                      help="HuggingFace model repository")
    parser.add_argument("-m", "--model-file", default="model.onnx",
                      help="Model filename")
    parser.add_argument("-t", "--tags-file", default="selected_tags.csv",
                      help="Tags filename")
    args = parser.parse_args()

    total_start = time.time()

    # Download or use existing files
    if not os.path.exists(args.model_file) or not os.path.exists(args.tags_file):
        print("Downloading model files...")
        if not os.path.exists(args.model_file):
            download_file(args.model_repo, args.model_file, args.model_file)
        if not os.path.exists(args.tags_file):
            download_file(args.model_repo, "selected_tags.csv", args.tags_file)

    print("Initializing tagger...")
    tagger = Tagger(args.model_file, args.tags_file)

    print("Processing image...")
    process_start = time.time()
    image = Image.open(args.image)
    general, characters, ratings = tagger.predict(image)
    print(f"Total processing time: {time.time() - process_start:.2f}s")

    print("\nRatings:")
    for name, conf in ratings:
        print(f"{name}: {conf*100:.1f}%")

    print(f"\nGeneral tags ({len(general)}):")
    for name, conf in general:
        print(f"{name}: {conf*100:.1f}%")

    print(f"\nCharacter tags ({len(characters)}):")
    for name, conf in characters:
        print(f"{name}: {conf*100:.1f}%")

    print(f"\nTotal execution time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()