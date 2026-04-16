from pathlib import Path
import pandas as pd


def find_nifti_files(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    return sorted(
        [
            p for p in root_dir.rglob("*")
            if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))
        ]
    )


def classify_nifti_file(file_path: Path) -> str:
    name = file_path.name.lower()
    parent = file_path.parent.name.lower()
    full_path = str(file_path).lower()

    mask_keywords = ["mask", "label", "labels", "seg", "segmentation", "annot"]
    image_keywords = ["image", "img", "scan", "mr", "mri"]

    if any(k in name for k in mask_keywords):
        return "mask_like"
    if any(k in parent for k in mask_keywords):
        return "mask_like"
    if any(k in full_path for k in ["labelstr", "masks", "segmentations", "annotations"]):
        return "mask_like"

    if any(k in name for k in image_keywords):
        return "image_like"

    return "unknown"


def build_mask_availability_table(root_dir: str | Path) -> pd.DataFrame:
    files = find_nifti_files(root_dir)

    rows = []
    for f in files:
        rows.append(
            {
                "file_name": f.name,
                "relative_path": str(f.relative_to(root_dir)),
                "parent_folder": f.parent.name,
                "file_type_guess": classify_nifti_file(f),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = Path("PASTE_ROOT_PATH_HERE")
    df = build_mask_availability_table(root)

    print("Total NIfTI files:", len(df))
    print("\nCounts by guessed type:")
    print(df["file_type_guess"].value_counts(dropna=False))

    print("\nFirst 30 files:")
    print(df.head(30).to_string(index=False))