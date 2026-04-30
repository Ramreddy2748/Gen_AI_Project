import os
import shutil
from zipfile import ZipFile

sample_dir = "data/sample_videos"
adl_dir = "data/adl"
falls_dir = "data/falls"

# Ensure directories exist
os.makedirs(adl_dir, exist_ok=True)
os.makedirs(falls_dir, exist_ok=True)

for filename in os.listdir(sample_dir):
    if filename.endswith(".zip"):
        src_path = os.path.join(sample_dir, filename)
        if filename.startswith("adl-"):
            dest_path = os.path.join(adl_dir, filename)
            shutil.move(src_path, dest_path)
            # Extract
            with ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(adl_dir)
            os.remove(dest_path)
            print(f"Extracted ADL: {filename}")
        elif filename.startswith("fall-"):
            # Extract number
            parts = filename.split("-")
            num = parts[1]
            seq_dir = os.path.join(falls_dir, f"fall-{num}")
            os.makedirs(seq_dir, exist_ok=True)
            dest_path = os.path.join(seq_dir, filename)
            shutil.move(src_path, dest_path)
            # Extract
            with ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(seq_dir)
            os.remove(dest_path)
            print(f"Extracted Fall: {filename}")

print("All zips extracted and organized.")