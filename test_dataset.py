#!/usr/bin/env python3
"""
Test script to download and explore the Kaggle dataset
"""

import kagglehub
import os

def test_dataset_download():
    """Test the CREMA-D dataset download functionality"""
    print("🧪 Testing CREMA-D Dataset Download")
    print("=" * 40)
    
    try:
        print("📥 Downloading CREMA-D dataset...")
        path = kagglehub.dataset_download("ejlok1/cremad")
        
        print(f"✅ Dataset downloaded to: {path}")
        print(f"📁 Path exists: {os.path.exists(path)}")
        
        if os.path.exists(path):
            print(f"📊 Directory contents:")
            files = os.listdir(path)
            print(f"  Total files/folders: {len(files)}")
            
            for item in files:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                    print(f"  📄 {item} ({size:.2f} MB)")
                elif os.path.isdir(item_path):
                    subitems = os.listdir(item_path)
                    print(f"  📂 {item}/ ({len(subitems)} items)")
                    
                    # Show first few items in subdirectories
                    for subitem in subitems[:3]:
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isfile(subitem_path):
                            size = os.path.getsize(subitem_path) / (1024 * 1024)
                            print(f"    📄 {subitem} ({size:.2f} MB)")
                        else:
                            print(f"    📂 {subitem}/")
                    
                    if len(subitems) > 3:
                        print(f"    ... and {len(subitems) - 3} more items")
        
        return path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Install kagglehub: pip install kagglehub")
        print("2. Login to Kaggle: kagglehub login")
        print("3. Check your internet connection")
        return None

if __name__ == "__main__":
    test_dataset_download() 