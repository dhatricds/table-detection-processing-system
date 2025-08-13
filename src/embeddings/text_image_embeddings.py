# Text and Image Embedding Generation with FAISS Indexing
import os
import faiss
import cv2
import numpy as np
import pytesseract
import openai
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import json
import pandas as pd
from PIL import Image

# Configuration - Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# File paths
META_JSON    = "output/final_columns.json"
TEXT_INDEX   = "output/text_index.faiss"
TEXT_MAP_CSV = "output/text_map.csv"
IMG_INDEX    = "output/image_index.faiss"
IMG_MAP_CSV  = "output/image_map.csv"

def load_metadata(meta_json_path):
    """Load metadata from JSON file"""
    return json.loads(Path(meta_json_path).read_text())

def create_text_embeddings(records, openai_api_key):
    """
    Create text embeddings using OpenAI's text-embedding-ada-002 model
    """
    print("🔄 Creating text embeddings...")
    
    # Prepare text data
    texts, text_ids = [], []
    for i, r in enumerate(records):
        parts = [r.get("description",""), r.get("mounting_height",""), r.get("notes","")]
        txt = " | ".join([p for p in parts if p])
        if txt:
            texts.append(txt)
            text_ids.append(i)

    print(f"📝 Processing {len(texts)} text entries...")

    # Initialize OpenAI client
    openai.api_key = openai_api_key
    
    # Embed in batches
    text_embs = []
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        resp = openai.embeddings.create(model="text-embedding-ada-002", input=batch)
        text_embs += [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    return text_embs, text_ids, texts

def create_image_embeddings(records):
    """
    Create image embeddings using CLIP model with rotation augmentation
    """
    print("🔄 Creating image embeddings with rotation augmentation...")
    
    # Load CLIP model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    img_embs, img_ids = [], []
    angles = [0, 90, 180, 270]  # Rotation augmentation
    
    for i, r in enumerate(records):
        sym_path = r.get("symbol_image","")
        if sym_path and Path(sym_path).exists():
            try:
                base = Image.open(sym_path).convert("RGB")
                
                # Create embeddings for each rotation
                for angle in angles:
                    rot = base.rotate(angle, expand=True)
                    inp = processor(images=rot, return_tensors="pt")
                    
                    with torch.no_grad():
                        feat = model.get_image_features(**inp)[0].cpu().numpy().astype(np.float32)
                    
                    img_embs.append(feat)
                    img_ids.append(i)
                    
            except Exception as e:
                print(f"⚠️  Error processing image {sym_path}: {e}")
                continue

    print(f"🖼️  Created {len(img_embs)} image embeddings (with rotations)")
    return img_embs, img_ids

def build_faiss_index(embeddings, index_path, map_csv_path, record_ids, additional_data=None):
    """
    Build and save FAISS index with mapping
    """
    if not embeddings:
        print("⚠️  No embeddings to index")
        return
    
    print(f"🔧 Building FAISS index with {len(embeddings)} embeddings...")
    
    # Build index
    d = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(d)
    index.add(np.vstack(embeddings))
    
    # Save index
    faiss.write_index(index, index_path)
    print(f"💾 Saved FAISS index to {index_path}")
    
    # Save mapping
    if additional_data:
        # For image embeddings with additional metadata
        df = pd.DataFrame({
            "record_idx": record_ids,
            **additional_data
        })
    else:
        # For text embeddings
        df = pd.DataFrame({
            "record_idx": record_ids
        })
    
    df.to_csv(map_csv_path, index=False)
    print(f"💾 Saved mapping to {map_csv_path}")

def main():
    """
    Main function to create text and image embeddings
    """
    print("🚀 Starting embedding generation...")
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load metadata
    try:
        records = load_metadata(META_JSON)
        print(f"📊 Loaded {len(records)} records from {META_JSON}")
    except FileNotFoundError:
        print(f"❌ Error: Metadata file {META_JSON} not found")
        return
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Text Embeddings
    try:
        text_embs, text_ids, texts = create_text_embeddings(records, openai_api_key)
        build_faiss_index(text_embs, TEXT_INDEX, TEXT_MAP_CSV, text_ids)
    except Exception as e:
        print(f"❌ Error creating text embeddings: {e}")
        return
    
    # 2. Image Embeddings
    try:
        img_embs, img_ids = create_image_embeddings(records)
        if img_embs:
            # Prepare additional data for image mapping
            additional_data = {
                "symbol_image": [records[i]["symbol_image"] for i in img_ids]
            }
            build_faiss_index(img_embs, IMG_INDEX, IMG_MAP_CSV, img_ids, additional_data)
        else:
            print("⚠️  No image embeddings created")
    except Exception as e:
        print(f"❌ Error creating image embeddings: {e}")
        return
    
    print("✅ Successfully created text & image FAISS indices with rotation augmentation!")
    print(f"📁 Output files:")
    print(f"   - Text index: {TEXT_INDEX}")
    print(f"   - Text mapping: {TEXT_MAP_CSV}")
    print(f"   - Image index: {IMG_INDEX}")
    print(f"   - Image mapping: {IMG_MAP_CSV}")

if __name__ == "__main__":
    main()