from pathlib import Path
import numpy as np
import pandas as pd

from src.dataio.load_nifti import load_nifti, get_nifti_spacing, safe_stem


def find_nifti_files(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)

    all_files = [p for p in root_dir.rglob("*") if p.is_file()]
    nii_files = [
        p for p in all_files
        if p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")
    ]
    return sorted(nii_files)


def extract_scan_features(nifti_path: str | Path, root_dir: str | Path) -> dict:
    nifti_path = Path(nifti_path)
    root_dir = Path(root_dir)

    data, affine, header = load_nifti(nifti_path)
    spacing = tuple(float(x) for x in header.get_zooms())

    shape = data.shape
    ndim = data.ndim

    shape_x = shape[0] if ndim >= 1 else None
    shape_y = shape[1] if ndim >= 2 else None
    shape_z = shape[2] if ndim >= 3 else None

    spacing_x = spacing[0] if len(spacing) >= 1 else None
    spacing_y = spacing[1] if len(spacing) >= 2 else None
    spacing_z = spacing[2] if len(spacing) >= 3 else None

    fov_x_mm = shape_x * spacing_x if shape_x is not None and spacing_x is not None else None
    fov_y_mm = shape_y * spacing_y if shape_y is not None and spacing_y is not None else None
    fov_z_mm = shape_z * spacing_z if shape_z is not None and spacing_z is not None else None

    flat = data.astype(np.float32).ravel()
    nonzero_fraction = float(np.count_nonzero(flat) / flat.size) if flat.size > 0 else None

    rel_path = nifti_path.relative_to(root_dir)

    return {
        "scan_id": safe_stem(nifti_path),
        "file_name": nifti_path.name,
        "relative_path": str(rel_path),
        "parent_folder": nifti_path.parent.name,
        "suffix": "".join(nifti_path.suffixes),
        "num_dimensions": int(ndim),
        "shape": str(shape),
        "shape_x": shape_x,
        "shape_y": shape_y,
        "shape_z": shape_z,
        "spacing": str(spacing),
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "spacing_z": spacing_z,
        "fov_x_mm": fov_x_mm,
        "fov_y_mm": fov_y_mm,
        "fov_z_mm": fov_z_mm,
        "num_voxels": int(flat.size),
        "dtype": str(data.dtype),
        "min_intensity": float(np.min(flat)),
        "max_intensity": float(np.max(flat)),
        "mean_intensity": float(np.mean(flat)),
        "std_intensity": float(np.std(flat)),
        "median_intensity": float(np.median(flat)),
        "p01_intensity": float(np.percentile(flat, 1)),
        "p05_intensity": float(np.percentile(flat, 5)),
        "p95_intensity": float(np.percentile(flat, 95)),
        "p99_intensity": float(np.percentile(flat, 99)),
        "nonzero_fraction": nonzero_fraction,
    }


def build_scan_inventory(root_dir: str | Path) -> pd.DataFrame:
    root_dir = Path(root_dir)
    files = find_nifti_files(root_dir)

    rows = []
    for file_path in files:
        try:
            row = extract_scan_features(file_path, root_dir)
            row["error"] = None
            rows.append(row)
        except Exception as e:
            rows.append(
                {
                    "scan_id": safe_stem(file_path),
                    "file_name": file_path.name,
                    "relative_path": str(file_path),
                    "parent_folder": file_path.parent.name,
                    "suffix": "".join(file_path.suffixes),
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)


def save_scan_inventory(root_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    df = build_scan_inventory(root_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df