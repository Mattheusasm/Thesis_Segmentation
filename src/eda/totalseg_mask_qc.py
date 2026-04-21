from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.mask_level_eda import load_multilabel_mask, get_spacing


REQUIRED_TOTAL_MR_ORGANS = {
    "spleen": [1],
    "kidney_right": [2],
    "kidney_left": [3],
    "stomach": [6],
    "small_bowel": [13],
    "duodenum": [14],
    "colon": [15],
}


def _label_span(mask, label_ids, axis=2):
    label_ids = [int(x) for x in label_ids]
    binary = np.isin(mask, label_ids)

    if axis == 0:
        scores = binary.sum(axis=(1, 2))
    elif axis == 1:
        scores = binary.sum(axis=(0, 2))
    elif axis == 2:
        scores = binary.sum(axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    present_slices = np.where(scores > 0)[0]

    if len(present_slices) == 0:
        return {
            "first_slice": None,
            "last_slice": None,
            "n_slices_present": 0,
            "best_slice": None,
        }

    return {
        "first_slice": int(present_slices.min()),
        "last_slice": int(present_slices.max()),
        "n_slices_present": int(len(present_slices)),
        "best_slice": int(np.argmax(scores)),
    }


def check_one_mask_file(mask_path, required_organs=REQUIRED_TOTAL_MR_ORGANS):
    """
    Check whether one TotalSegmentator multilabel mask contains all required organs.
    """
    mask_path = Path(mask_path)

    mask, img, label_map = load_multilabel_mask(mask_path)
    spacing_x, spacing_y, spacing_z = get_spacing(img)
    voxel_volume_mm3 = spacing_x * spacing_y * spacing_z

    rows = []

    for organ_name, label_ids in required_organs.items():
        organ_binary = np.isin(mask, label_ids)

        voxel_count = int(organ_binary.sum())
        present = voxel_count > 0

        span = _label_span(mask, label_ids, axis=2)

        rows.append(
            {
                "file_name": mask_path.name,
                "organ_name": organ_name,
                "label_ids": ",".join(str(x) for x in label_ids),
                "present": present,
                "voxel_count": voxel_count,
                "volume_mm3": voxel_count * voxel_volume_mm3,
                "volume_ml": (voxel_count * voxel_volume_mm3) / 1000.0,
                **span,
            }
        )

    return pd.DataFrame(rows)


def check_all_scans_have_required_masks(
    scan_root,
    mask_root,
    required_organs=REQUIRED_TOTAL_MR_ORGANS,
):
    """
    Strict dataset-level check.

    For every scan in scan_root:
    - check if a matching mask exists in mask_root
    - if it exists, check all required organs
    """
    scan_root = Path(scan_root)
    mask_root = Path(mask_root)

    scan_paths = sorted(scan_root.glob("*.nii.gz"))
    mask_paths = sorted(mask_root.glob("*.nii.gz"))

    if len(scan_paths) == 0:
        raise FileNotFoundError(f"No scan .nii.gz files found in: {scan_root}")

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No mask .nii.gz files found in: {mask_root}")

    all_long_rows = []
    all_summary_rows = []

    mask_names = {p.name for p in mask_paths}
    scan_names = {p.name for p in scan_paths}

    extra_mask_names = sorted(mask_names - scan_names)

    for scan_path in scan_paths:
        mask_path = mask_root / scan_path.name
        mask_exists = mask_path.exists()

        if not mask_exists:
            missing_organs = list(required_organs.keys())

            all_summary_rows.append(
                {
                    "file_name": scan_path.name,
                    "scan_exists": True,
                    "mask_exists": False,
                    "all_required_organs_present": False,
                    "n_missing_organs": len(missing_organs),
                    "missing_organs": ",".join(missing_organs),
                }
            )

            for organ_name, label_ids in required_organs.items():
                all_long_rows.append(
                    {
                        "file_name": scan_path.name,
                        "organ_name": organ_name,
                        "label_ids": ",".join(str(x) for x in label_ids),
                        "present": False,
                        "voxel_count": 0,
                        "volume_mm3": 0.0,
                        "volume_ml": 0.0,
                        "first_slice": None,
                        "last_slice": None,
                        "n_slices_present": 0,
                        "best_slice": None,
                    }
                )

            continue

        organ_df = check_one_mask_file(mask_path, required_organs=required_organs)

        missing_organs = organ_df.loc[~organ_df["present"], "organ_name"].tolist()

        all_summary_rows.append(
            {
                "file_name": scan_path.name,
                "scan_exists": True,
                "mask_exists": True,
                "all_required_organs_present": len(missing_organs) == 0,
                "n_missing_organs": len(missing_organs),
                "missing_organs": ",".join(missing_organs),
            }
        )

        all_long_rows.append(organ_df)

    summary_df = pd.DataFrame(all_summary_rows)

    long_parts = []
    for item in all_long_rows:
        if isinstance(item, pd.DataFrame):
            long_parts.append(item)
        else:
            long_parts.append(pd.DataFrame([item]))

    organ_presence_df = pd.concat(long_parts, ignore_index=True)

    extra_masks_df = pd.DataFrame(
        {
            "extra_mask_file_name": extra_mask_names
        }
    )

    return summary_df, organ_presence_df, extra_masks_df


def print_qc_result(summary_df, organ_presence_df, extra_masks_df):
    """
    Print a strict readable QC result.
    """
    n_scans = len(summary_df)
    n_masks_missing = int((~summary_df["mask_exists"]).sum())
    n_failed_organs = int((~organ_presence_df["present"]).sum())
    n_files_missing_organs = int((~summary_df["all_required_organs_present"]).sum())

    print("STRICT TOTAL-SEGMENTATOR MASK QC")
    print("--------------------------------")
    print(f"Number of scans checked: {n_scans}")
    print(f"Scans without matching mask: {n_masks_missing}")
    print(f"Files missing at least one required organ: {n_files_missing_organs}")
    print(f"Total missing organ entries: {n_failed_organs}")
    print(f"Extra mask files without matching scan: {len(extra_masks_df)}")

    if n_masks_missing == 0 and n_failed_organs == 0:
        print("\nPASS: every scan has a mask and every required organ is present.")
    else:
        print("\nFAIL: some scans/masks are incomplete. Inspect the tables below.")