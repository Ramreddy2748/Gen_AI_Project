"""
Download GMNCSA24 Dataset from GitHub

Downloads videos directly from the GitHub repository:
https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
"""

import os
import subprocess
from pathlib import Path

def download_gmncsa24():
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "Dataset" / "GMNCSA24"
    TEMP_DIR = PROJECT_ROOT / "temp_gmncsa24"
    
    print("=" * 60)
    print("DOWNLOADING GMNCSA24 FROM GITHUB")
    print("=" * 60)
    
    # Clone the repository
    repo_url = "https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos.git"
    
    if TEMP_DIR.exists():
        print(f"Removing existing temp directory...")
        import shutil
        shutil.rmtree(TEMP_DIR)
    
    print(f"\nCloning repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(TEMP_DIR)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Error cloning: {result.stderr}")
        return False
    
    print("Repository cloned successfully")
    
    # Find video directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fall_dir = OUTPUT_DIR / "Fall"
    nofall_dir = OUTPUT_DIR / "No-Fall"
    fall_dir.mkdir(exist_ok=True)
    nofall_dir.mkdir(exist_ok=True)
    
    import shutil
    
    # Look for Fall and ADL directories
    fall_count = 0
    nofall_count = 0
    
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI')):
                src_path = Path(root) / file
                
                # Determine label from path or filename
                path_lower = str(src_path).lower()
                if 'fall' in path_lower and 'adl' not in path_lower:
                    fall_count += 1
                    dst_path = fall_dir / f"gmncsa_fall_{fall_count:03d}{src_path.suffix}"
                    print(f"  Fall: {file}")
                else:
                    nofall_count += 1
                    dst_path = nofall_dir / f"gmncsa_adl_{nofall_count:03d}{src_path.suffix}"
                    print(f"  ADL: {file}")
                
                shutil.copy(src_path, dst_path)
    
    # Cleanup
    print("\nCleaning up...")
    shutil.rmtree(TEMP_DIR)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nFall videos: {fall_count}")
    print(f"No-Fall (ADL) videos: {nofall_count}")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    download_gmncsa24()
