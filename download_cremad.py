#!/usr/bin/env python3
"""
Simple script to download the CREMA-D dataset from Kaggle
"""

import kagglehub
import os

def download_cremad_dataset():
    """Download the CREMA-D emotional speech dataset from Kaggle"""
    print("=== Downloading CREMA-D Dataset ===")
    
    try:
        # Download latest version of the CREMA-D dataset
        path = kagglehub.dataset_download("ejlok1/cremad")
        print(f"✅ CREMA-D dataset downloaded successfully to: {path}")
        
        # List contents
        if os.path.exists(path):
            print(f"\n📁 Dataset contents:")
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                    print(f"  📄 {item} ({size:.2f} MB)")
                elif os.path.isdir(item_path):
                    subitems = os.listdir(item_path)
                    print(f"  📂 {item}/ ({len(subitems)} items)")
        
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please ensure you have kagglehub installed and configured:")
        print("pip install kagglehub")
        return None

if __name__ == "__main__":
    dataset_path = download_cremad_dataset()
    if dataset_path:
        print(f"\n🎉 Dataset ready at: {dataset_path}")
    else:
        print("\n💥 Failed to download dataset")
