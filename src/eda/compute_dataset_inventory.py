from pathlib import Path
import pandas as pd
import numpy as np

from src.dataio.load_nifti import load_nifti, get_nifti_shape, get_nifti_spacing


def find_nifti_files(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    nii_files = list(root_dir.rglob("*.nii")) + list(root_dir.rglob("*.nii.gz"))
    return sorted(nii_files)


def build_dataset_inventory(root_dir: str | Path) -> pd.DataFrame:
    files = find_nifti_files(root_dir)

    columns = [
        "file_path",
        "file_name",
        "parent_folder",
        "shape",
        "spacing",
        "num_dimensions",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
        "error",
    ]

    rows = []
    for file_path in files:
        try:
            data, _, _ = load_nifti(file_path)
            shape = get_nifti_shape(file_path)
            spacing = get_nifti_spacing(file_path)

            rows.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "parent_folder": file_path.parent.name,
                    "shape": str(shape),
                    "spacing": str(spacing),
                    "num_dimensions": len(shape),
                    "dtype": str(data.dtype),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "error": None,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "parent_folder": file_path.parent.name,
                    "shape": None,
                    "spacing": None,
                    "num_dimensions": None,
                    "dtype": None,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows, columns=columns)