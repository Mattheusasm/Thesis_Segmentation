from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_nifti(nifti_path: str | Path) -> tuple[np.ndarray, np.ndarray, Any]:
    nifti_path = Path(nifti_path)

    if not nifti_path.exists():
        raise FileNotFoundError(f"File not found: {nifti_path}")

    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    return data, img.affine, img.header


def get_nifti_shape(nifti_path: str | Path) -> tuple[int, ...]:
    data, _, _ = load_nifti(nifti_path)
    return data.shape


def get_nifti_spacing(nifti_path: str | Path) -> tuple[float, ...]:
    nifti_path = Path(nifti_path)

    if not nifti_path.exists():
        raise FileNotFoundError(f"File not found: {nifti_path}")

    img = nib.load(str(nifti_path))
    spacing = img.header.get_zooms()

    return tuple(float(x) for x in spacing)


def summarize_nifti(nifti_path: str | Path) -> dict:
    data, _, header = load_nifti(nifti_path)

    return {
        "path": str(nifti_path),
        "shape": data.shape,
        "dtype": str(data.dtype),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "spacing": tuple(float(x) for x in header.get_zooms()),
    }


def safe_stem(path: Path) -> str:
    """
    Handles both .nii and .nii.gz correctly.
    """
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem