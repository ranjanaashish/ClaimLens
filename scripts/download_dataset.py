"""
Download INS-MMBench dataset from Hugging Face.
Repo: https://huggingface.co/datasets/FDU-INS/INS-MMBench
"""
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_REPO = "FDU-INS/INS-MMBench"
FILES = [
    ("multi_step_claim.tsv", "35 MB - Multi-step claim scenario (auto insurance)"),
    ("multi_step_agri.tsv", "44 MB - Multi-step agricultural insurance"),
    ("multi_step_health.tsv", "188 MB - Multi-step health insurance"),
    ("multi_step_liability.tsv", "260 MB - Multi-step liability insurance"),
    ("multi_step_property.tsv", "767 MB - Multi-step property insurance"),
    ("INS_MMBench_fundamental.tsv", "1.47 GB - Fundamental multimodal insurance benchmark"),
]

def main():
    base_dir = Path(__file__).parent.parent
    local_dataset_dir = base_dir / "data" / "INS-MMBench" / "dataset"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Also prepare LMUData directory for VLMEvalKit compatibility
    lmu_data_dir = Path(os.environ.get("LMUData", Path.home() / "LMUData"))
    lmu_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target directories:")
    print(f"  - Local project: {local_dataset_dir}")
    print(f"  - LMUData (eval): {lmu_data_dir}")
    print()

    for filename, desc in FILES:
        target_file = local_dataset_dir / filename
        lmu_file = lmu_data_dir / filename
        
        if target_file.exists() and target_file.stat().st_size > 1024 * 1024:
            print(f"[*] Already exists: {filename} ({target_file.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"[+] Downloading: {filename} ({desc})...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(local_dataset_dir),
                    resume_download=True,
                )
                print(f"    Downloaded to: {downloaded_path}")
            except Exception as e:
                print(f"[!] Error downloading {filename}: {e}")
                continue

        # Ensure copy/link in LMUData
        if target_file.exists():
            if not lmu_file.exists():
                try:
                    # Try creating hard link first (instant, 0 disk space)
                    os.link(str(target_file), str(lmu_file))
                    print(f"    Linked to LMUData: {lmu_file}")
                except Exception:
                    # Fallback to copy
                    shutil.copyfile(str(target_file), str(lmu_file))
                    print(f"    Copied to LMUData: {lmu_file}")
            else:
                print(f"    Available in LMUData: {lmu_file}")
        print()

    print("=== All downloads complete ===")

if __name__ == "__main__":
    main()
