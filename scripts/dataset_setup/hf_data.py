from pathlib import Path

from datasets import load_dataset

DATASET_ID = "ud-smart-city/fall-detection"
OUTPUT_DIR = Path("data/hf_fall_detection")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)

    print("\nDataset splits:")
    for split_name, split_ds in ds.items():
        print(f"- {split_name}: {len(split_ds)} rows")

    save_path = OUTPUT_DIR / "dataset"
    ds.save_to_disk(str(save_path))
    print(f"\nSaved Hugging Face dataset to: {save_path}")


if __name__ == "__main__":
    main()