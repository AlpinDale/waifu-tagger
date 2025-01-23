use anyhow::{anyhow, Result};
use image::{DynamicImage, GenericImageView};
use reqwest;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tract_ndarray as ndarray;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;

const MODEL_REPO: &str = "SmilingWolf/wd-swinv2-tagger-v3";
const MODEL_FILENAME: &str = "model.onnx";
const LABELS_FILENAME: &str = "selected_tags.csv";

async fn download_file(repo: &str, filename: &str, output_path: &PathBuf) -> Result<()> {
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
    println!("Downloading from: {}", url);
    let response = reqwest::get(&url).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download {}: {}",
            filename,
            response.status()
        ));
    }

    let bytes = response.bytes().await?;
    println!("Downloaded {} bytes", bytes.len());
    fs::write(output_path, bytes)?;
    println!("Saved to: {}", output_path.display());

    Ok(())
}

#[derive(Debug)]
struct TagWithConfidence {
    name: String,
    confidence: f32,
}

struct Tagger {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    tag_names: Vec<String>,
    rating_indexes: Vec<usize>,
    general_indexes: Vec<usize>,
    character_indexes: Vec<usize>,
    model_target_size: usize,
}

impl Tagger {
    pub fn new(model_path: PathBuf, labels_path: PathBuf) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(&model_path)?
            .into_optimized()?
            .into_runnable()?;

        println!("Model loaded. Input facts:");
        let input_fact = model.model().input_fact(0)?;
        println!("Input 0: {:?}", input_fact);
        println!("Input shape: {:?}", input_fact.shape);

        let dims = input_fact.shape.dims();
        println!("Dimensions: {:?}", dims);

        let model_target_size = dims[1]
            .to_usize()
            .map_err(|e| anyhow!("Could not determine target size: {}", e))?;

        println!("Target size: {}", model_target_size);

        println!("Loading labels from {:?}", labels_path);
        let mut reader = csv::Reader::from_path(labels_path)?;
        let mut tag_names = Vec::new();
        let mut rating_indexes = Vec::new();
        let mut general_indexes = Vec::new();
        let mut character_indexes = Vec::new();
        let mut id_to_index = HashMap::new();

        for (idx, record) in reader.records().enumerate() {
            let record = record?;
            let tag_id: usize = record
                .get(0)
                .ok_or_else(|| anyhow!("Missing tag_id column"))?
                .parse()?;
            let name = record
                .get(1)
                .ok_or_else(|| anyhow!("Missing name column"))?
                .to_string();
            let category: i32 = record
                .get(2)
                .ok_or_else(|| anyhow!("Missing category column"))?
                .parse()?;

            // println!("Loaded tag: id={}, name={}, category={}", tag_id, name, category);

            tag_names.push(name);
            id_to_index.insert(tag_id, idx);

            match category {
                9 => rating_indexes.push(idx),
                0 => general_indexes.push(idx),
                4 => character_indexes.push(idx),
                _ => {}
            }
        }

        // println!("Loaded {} tags", tag_names.len());
        // println!("First few tags: {:?}", &tag_names[..5]);

        Ok(Self {
            model,
            tag_names,
            rating_indexes,
            general_indexes,
            character_indexes,
            model_target_size,
        })
    }

    pub async fn download_model() -> Result<Self> {
        let model_path = PathBuf::from(MODEL_FILENAME);
        let labels_path = PathBuf::from(LABELS_FILENAME);

        if !model_path.exists() {
            println!("Downloading model...");
            download_file(MODEL_REPO, MODEL_FILENAME, &model_path).await?;
        }

        if !labels_path.exists() {
            println!("Downloading labels...");
            download_file(MODEL_REPO, LABELS_FILENAME, &labels_path).await?;
        }

        Self::new(model_path, labels_path)
    }

    pub fn prepare_image(&self, image: DynamicImage) -> Result<Tensor> {
        let target_size = self.model_target_size;

        let image = image.to_rgb8();

        let (width, height) = image.dimensions();
        let max_dim = width.max(height);

        let mut padded_image = image::RgbImage::new(max_dim, max_dim);
        for pixel in padded_image.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }

        let pad_left = (max_dim - width) / 2;
        let pad_top = (max_dim - height) / 2;

        image::imageops::replace(&mut padded_image, &image, pad_left.into(), pad_top.into());

        let padded_image = if max_dim != target_size as u32 {
            image::imageops::resize(
                &padded_image,
                target_size as u32,
                target_size as u32,
                image::imageops::FilterType::Triangle,
            )
        } else {
            padded_image
        };

        let array =
            ndarray::Array4::from_shape_fn((1, target_size, target_size, 3), |(_, y, x, c)| {
                let pixel = padded_image.get_pixel(x as u32, y as u32);
                let c = 2 - c; // Reverse channel order
                pixel[c] as f32
            });

        Ok(array.into())
    }

    pub fn predict(
        &self,
        image: DynamicImage,
        general_threshold: f32,
        character_threshold: f32,
    ) -> Result<(
        Vec<TagWithConfidence>,
        Vec<TagWithConfidence>,
        Vec<TagWithConfidence>,
    )> {
        let input = self.prepare_image(image)?;
        let result = self.model.run(tvec!(input.into()))?;
        let output = result[0].to_array_view::<f32>()?;
        let predictions = output.as_slice().unwrap();

        let mut general_tags = Vec::new();
        let mut character_tags = Vec::new();
        let mut rating_tags = Vec::new();

        for &idx in &self.rating_indexes {
            rating_tags.push(TagWithConfidence {
                name: self.tag_names[idx].clone(),
                confidence: predictions[idx],
            });
        }
        rating_tags.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        for &idx in &self.general_indexes {
            if predictions[idx] > general_threshold {
                general_tags.push(TagWithConfidence {
                    name: self.tag_names[idx].clone(),
                    confidence: predictions[idx],
                });
            }
        }

        for &idx in &self.character_indexes {
            if predictions[idx] > character_threshold {
                character_tags.push(TagWithConfidence {
                    name: self.tag_names[idx].clone(),
                    confidence: predictions[idx],
                });
            }
        }

        general_tags.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        character_tags.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok((general_tags, character_tags, rating_tags))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing tagger...");
    let tagger = Tagger::download_model().await?;

    let image_path = "test.png";
    if !std::path::Path::new(image_path).exists() {
        return Err(anyhow!("Test image not found: {}", image_path));
    }

    println!("Processing image...");
    let image = image::open(image_path)?;
    let (general_tags, character_tags, rating_tags) = tagger.predict(image, 0.35, 0.85)?;

    println!("\nRatings:");
    for tag in rating_tags {
        println!("{}: {:.1}%", tag.name, tag.confidence * 100.0);
    }

    println!("\nGeneral tags ({}):", general_tags.len());
    for tag in general_tags {
        println!("{}: {:.1}%", tag.name, tag.confidence * 100.0);
    }

    println!("\nCharacter tags ({}):", character_tags.len());
    for tag in character_tags {
        println!("{}: {:.1}%", tag.name, tag.confidence * 100.0);
    }

    Ok(())
}
