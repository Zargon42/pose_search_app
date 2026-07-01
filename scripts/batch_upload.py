#!/usr/bin/env python3
"""
Batch upload images from a folder to the Pose Search App.
Checks for duplicate poses based on vector similarity.

Usage:
    python batch_upload.py <folder_path> [--similarity-threshold 0.05] [--api-url http://localhost:8000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Add parent directory to path to import mediapipe
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import mediapipe as mp
from main import normalize_pose, extract_pose_vector


def load_current_index(api_url):
    """Load the current pose index from the server."""
    try:
        data_dir = Path(__file__).parent.parent / "app" / "data"
        index_path = data_dir / "pose_index.npz"
        db_path = data_dir / "db_files.json"
        
        if not index_path.exists() or not db_path.exists():
            return None, []
        
        index_data = np.load(index_path)
        pose_index = index_data["vectors"]
        
        with open(db_path, "r", encoding="utf-8") as f:
            file_paths = json.load(f)
        
        return pose_index, file_paths
    except Exception as e:
        print(f"Error loading index: {e}")
        return None, []


def is_duplicate(vector, existing_vectors, threshold=0.05):
    """
    Check if a vector is a duplicate of existing vectors.
    Returns (is_duplicate, min_distance, closest_idx)
    """
    if existing_vectors is None or len(existing_vectors) == 0:
        return False, float('inf'), -1
    
    distances = np.linalg.norm(existing_vectors - vector, axis=1)
    min_distance = np.min(distances)
    closest_idx = int(np.argmin(distances))
    
    return min_distance < threshold, min_distance, closest_idx


def batch_upload(folder_path, similarity_threshold=0.05, api_url="http://localhost:8000"):
    """Upload images from a folder, skipping duplicates."""
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in '{folder_path}'.")
        return
    
    print(f"Found {len(image_files)} image(s) to process.")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"API URL: {api_url}\n")
    
    # Load current index
    print("Loading current index...")
    pose_index, file_paths = load_current_index(api_url)
    print(f"Current index has {len(file_paths)} image(s).\n")
    
    uploaded = 0
    duplicates = 0
    failed = 0
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}...", end=" ")
        
        try:
            # Open and extract pose
            image = Image.open(image_path)
            vector = extract_pose_vector(image)
            
            if vector is None:
                print("FAILED - No pose detected")
                failed += 1
                continue
            
            # Check for duplicates
            is_dup, distance, closest_idx = is_duplicate(vector, pose_index, similarity_threshold)
            
            if is_dup:
                closest_file = file_paths[closest_idx] if closest_idx >= 0 else "unknown"
                print(f"SKIPPED - Duplicate (distance: {distance:.4f}, similar to: {closest_file})")
                duplicates += 1
                continue
            
            # Upload to server
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/jpeg")}
                response = requests.post(f"{api_url}/api/upload", files=files, timeout=30)
            
            if response.status_code == 200:
                print(f"UPLOADED")
                uploaded += 1
                
                # Update local index for subsequent duplicate checks
                if pose_index is None:
                    pose_index = vector.reshape(1, -1)
                else:
                    pose_index = np.vstack([pose_index, vector])
                file_paths.append(image_path.name)
            else:
                error_data = response.json()
                error_msg = error_data.get("detail", "Unknown error")
                print(f"FAILED - {error_msg}")
                failed += 1
        
        except Exception as e:
            print(f"FAILED - {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH UPLOAD SUMMARY")
    print("=" * 50)
    print(f"Uploaded:   {uploaded}")
    print(f"Duplicates: {duplicates}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(image_files)}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch upload images to Pose Search App with duplicate detection"
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.05,
        help="Similarity threshold for duplicate detection (default: 0.05)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)",
    )
    
    args = parser.parse_args()
    
    batch_upload(args.folder, args.similarity_threshold, args.api_url)
