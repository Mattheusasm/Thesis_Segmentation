from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eda.mask_level_eda import load_multilabel_mask, get_spacing


SELECTED_TOTAL_MR_ORGANS = {
    "spleen": [1],
    "kidney_right": [2],
    "kidney_left": [3],
    "stomach": [6],
    "small_bowel": [13],
    "duodenum": [14],
    "colon": [15],
}


def _slice_span(mask, label_ids, axis=2):
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


def _bounding_box(mask, label_ids):
    label_ids = [int(x) for x in label_ids]
    binary = np.isin(mask, label_ids)
    coords = np.argwhere(binary)

    if coords.size == 0:
        return {
            "bbox_x_min": None,
            "bbox_x_max": None,
            "bbox_y_min": None,
            "bbox_y_max": None,
            "bbox_z_min": None,
            "bbox_z_max": None,
            "bbox_size_x": 0,
            "bbox_size_y": 0,
            "bbox_size_z": 0,
        }

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    return {
        "bbox_x_min": int(mins[0]),
        "bbox_x_max": int(maxs[0]),
        "bbox_y_min": int(mins[1]),
        "bbox_y_max": int(maxs[1]),
        "bbox_z_min": int(mins[2]),
        "bbox_z_max": int(maxs[2]),
        "bbox_size_x": int(maxs[0] - mins[0] + 1),
        "bbox_size_y": int(maxs[1] - mins[1] + 1),
        "bbox_size_z": int(maxs[2] - mins[2] + 1),
    }


def summarize_organs_in_one_mask(mask_path, selected_organs=SELECTED_TOTAL_MR_ORGANS):
    mask_path = Path(mask_path)

    mask, img, _ = load_multilabel_mask(mask_path)

    spacing_x, spacing_y, spacing_z = get_spacing(img)
    voxel_volume_mm3 = spacing_x * spacing_y * spacing_z
    total_voxels = int(mask.size)

    rows = []

    for organ_name, label_ids in selected_organs.items():
        organ_binary = np.isin(mask, label_ids)

        voxel_count = int(organ_binary.sum())
        present = voxel_count > 0

        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_ml = volume_mm3 / 1000.0

        span = _slice_span(mask, label_ids, axis=2)
        bbox = _bounding_box(mask, label_ids)

        rows.append(
            {
                "file_name": mask_path.name,
                "organ_name": organ_name,
                "label_ids": ",".join(str(x) for x in label_ids),
                "present": bool(present),
                "voxel_count": voxel_count,
                "volume_mm3": volume_mm3,
                "volume_ml": volume_ml,
                # Important:
                # For stats, missing organs become NaN, not 0.
                "volume_ml_for_stats": volume_ml if present else np.nan,
                "fraction_of_scan_voxels": voxel_count / total_voxels if total_voxels > 0 else 0.0,
                "shape_x": int(mask.shape[0]),
                "shape_y": int(mask.shape[1]),
                "shape_z": int(mask.shape[2]),
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "spacing_z": spacing_z,
                **span,
                **bbox,
            }
        )

    return pd.DataFrame(rows)


def summarize_organs_in_folder(mask_root, selected_organs=SELECTED_TOTAL_MR_ORGANS):
    mask_root = Path(mask_root)
    mask_paths = sorted(mask_root.glob("*.nii.gz"))

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No .nii.gz mask files found in: {mask_root}")

    all_rows = []

    for mask_path in mask_paths:
        one_df = summarize_organs_in_one_mask(
            mask_path=mask_path,
            selected_organs=selected_organs,
        )
        all_rows.append(one_df)

    organ_long_df = pd.concat(all_rows, ignore_index=True)

    return organ_long_df, mask_paths


def build_presence_counts(organ_long_df):
    out = (
        organ_long_df
        .groupby("organ_name")["present"]
        .agg(n_present="sum", n_total="count")
        .reset_index()
    )

    out["n_missing"] = out["n_total"] - out["n_present"]
    out["missing_fraction"] = out["n_missing"] / out["n_total"]

    return out.sort_values(["n_missing", "organ_name"], ascending=[False, True]).reset_index(drop=True)


def build_missing_organs_table(organ_long_df):
    return (
        organ_long_df[~organ_long_df["present"]]
        .copy()
        .sort_values(["organ_name", "file_name"])
        .reset_index(drop=True)
    )


def build_volume_stats(organ_long_df):
    present_df = organ_long_df[organ_long_df["present"]].copy()

    stats_df = (
        present_df
        .groupby("organ_name")["volume_ml_for_stats"]
        .agg(
            n_present="count",
            mean_ml="mean",
            std_ml="std",
            min_ml="min",
            median_ml="median",
            max_ml="max",
        )
        .reset_index()
    )

    stats_df = stats_df.sort_values("organ_name").reset_index(drop=True)

    return stats_df


def build_case_wide_summary(organ_long_df):
    volume_wide = organ_long_df.pivot(
        index="file_name",
        columns="organ_name",
        values="volume_ml_for_stats",
    )

    volume_wide.columns = [f"{col}_volume_ml" for col in volume_wide.columns]

    present_wide = organ_long_df.pivot(
        index="file_name",
        columns="organ_name",
        values="present",
    )

    present_wide.columns = [f"{col}_present" for col in present_wide.columns]

    out = pd.concat([volume_wide, present_wide], axis=1).reset_index()

    return out


def plot_missing_counts(presence_counts_df):
    plt.figure(figsize=(8, 4))
    plt.bar(presence_counts_df["organ_name"], presence_counts_df["n_missing"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of missing masks")
    plt.title("Missing organ masks per organ")
    plt.tight_layout()
    plt.show()


def plot_volume_boxplot(organ_long_df):
    present_df = organ_long_df[organ_long_df["present"]].copy()

    plt.figure(figsize=(10, 5))
    present_df.boxplot(column="volume_ml_for_stats", by="organ_name", rot=45)
    plt.suptitle("")
    plt.title("Organ volume distributions, missing organs excluded")
    plt.xlabel("Organ")
    plt.ylabel("Volume (mL)")
    plt.tight_layout()
    plt.show()


def plot_volume_histogram_for_organ(organ_long_df, organ_name):
    organ_df = organ_long_df[
        (organ_long_df["organ_name"] == organ_name)
        & (organ_long_df["present"])
    ].copy()

    if len(organ_df) == 0:
        print(f"No present masks found for organ: {organ_name}")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(organ_df["volume_ml_for_stats"], bins=20)
    plt.xlabel("Volume (mL)")
    plt.ylabel("Number of scans")
    plt.title(f"{organ_name} volume distribution")
    plt.tight_layout()
    plt.show()


def print_strict_eda_summary(organ_long_df, presence_counts_df, missing_organs_df):
    n_files = organ_long_df["file_name"].nunique()
    n_organs = organ_long_df["organ_name"].nunique()
    n_missing = len(missing_organs_df)

    print("STRICT ORGAN-LEVEL TOTAL-SEGMENTATOR EDA SUMMARY")
    print("------------------------------------------------")
    print(f"Number of mask files checked: {n_files}")
    print(f"Number of selected organs checked: {n_organs}")
    print(f"Total missing organ entries: {n_missing}")

    if n_missing == 0:
        print("\nPASS: all selected organs are present in all masks.")
    else:
        print("\nWARNING: some organ masks are missing.")
        print("These are treated as NOT SEGMENTED, not as true zero-volume anatomy.")
        print("\nMissing counts by organ:")
        print(presence_counts_df[["organ_name", "n_present", "n_total", "n_missing"]])



def build_volume_outlier_table(
    organ_long_df,
    organs=None,
    n_smallest=1,
    n_largest=1,
):
    """
    Build a table with smallest and largest non-missing organ masks.

    Missing organs are excluded from volume outlier selection because
    they are not true zero-volume anatomy.
    """
    if organs is None:
        organs = sorted(organ_long_df["organ_name"].unique())

    rows = []

    for organ_name in organs:
        organ_df = organ_long_df[
            (organ_long_df["organ_name"] == organ_name)
            & (organ_long_df["present"])
        ].copy()

        if len(organ_df) == 0:
            continue

        smallest_df = organ_df.sort_values("volume_ml_for_stats", ascending=True).head(n_smallest)
        largest_df = organ_df.sort_values("volume_ml_for_stats", ascending=False).head(n_largest)

        smallest_df = smallest_df.copy()
        smallest_df["outlier_type"] = "smallest_non_missing"

        largest_df = largest_df.copy()
        largest_df["outlier_type"] = "largest"

        rows.append(smallest_df)
        rows.append(largest_df)

    if len(rows) == 0:
        return pd.DataFrame()

    outlier_df = pd.concat(rows, ignore_index=True)

    wanted_cols = [
        "outlier_type",
        "file_name",
        "organ_name",
        "label_ids",
        "present",
        "voxel_count",
        "volume_ml",
        "volume_ml_for_stats",
        "first_slice",
        "last_slice",
        "n_slices_present",
        "best_slice",
    ]

    return outlier_df[wanted_cols].sort_values(
        ["organ_name", "outlier_type", "volume_ml_for_stats"]
    ).reset_index(drop=True)


def build_missing_qc_table(organ_long_df):
    """
    Table of missing organ predictions.
    """
    missing_df = organ_long_df[~organ_long_df["present"]].copy()

    wanted_cols = [
        "file_name",
        "organ_name",
        "label_ids",
        "present",
        "voxel_count",
        "volume_ml",
        "first_slice",
        "last_slice",
        "n_slices_present",
        "best_slice",
    ]

    return missing_df[wanted_cols].sort_values(
        ["organ_name", "file_name"]
    ).reset_index(drop=True)


def extract_organ_mask(mask, label_ids):
    """
    Keep only one organ or organ group.
    Everything else becomes 0.
    """
    label_ids = [int(x) for x in label_ids]
    return np.where(np.isin(mask, label_ids), mask, 0).astype(np.int32)


def label_ids_from_string(label_ids_string):
    """
    Convert '1,2,3' into [1, 2, 3].
    """
    return [int(x) for x in str(label_ids_string).split(",") if str(x).strip() != ""]


def print_outlier_qc_summary(outlier_df, missing_qc_df):
    print("STRICT VISUAL OUTLIER QC SELECTION")
    print("----------------------------------")
    print(f"Number of volume outlier rows selected: {len(outlier_df)}")
    print(f"Number of missing organ rows selected: {len(missing_qc_df)}")

    if len(outlier_df) > 0:
        print("\nSelected volume outliers:")
        print(
            outlier_df[
                [
                    "outlier_type",
                    "file_name",
                    "organ_name",
                    "volume_ml_for_stats",
                    "best_slice",
                ]
            ]
        )

    if len(missing_qc_df) > 0:
        print("\nMissing organ predictions:")
        print(
            missing_qc_df[
                [
                    "file_name",
                    "organ_name",
                    "label_ids",
                    "present",
                ]
            ]
        )

def create_initial_qc_decision_table(organ_long_df):
    """
    Create a QC table with one row per file-organ pair.
    Default: everything is included unless we flag it later.
    """
    qc_df = organ_long_df[
        [
            "file_name",
            "organ_name",
            "present",
            "voxel_count",
            "volume_ml",
            "volume_ml_for_stats",
            "n_slices_present",
            "best_slice",
        ]
    ].copy()

    qc_df["qc_flag"] = "ok"
    qc_df["qc_reason"] = ""
    qc_df["use_for_volume_stats"] = True

    # Missing organs are not true zero anatomy.
    qc_df.loc[~qc_df["present"], "qc_flag"] = "missing_prediction"
    qc_df.loc[~qc_df["present"], "qc_reason"] = "Organ was not segmented by TotalSegmentator."
    qc_df.loc[~qc_df["present"], "use_for_volume_stats"] = False

    return qc_df


def apply_manual_qc_flags(qc_df):
    """
    Apply strict manual flags based on visual outlier QC.
    You can edit this list later if visual inspection changes.
    """
    qc_df = qc_df.copy()

    manual_flags = [
        {
            "file_name": "case78_day22.nii.gz",
            "organ_name": "duodenum",
            "qc_flag": "exclude_near_empty_prediction",
            "qc_reason": "Duodenum prediction has only 4 voxels and is not reliable.",
            "use_for_volume_stats": False,
        },
        {
            "file_name": "case24_day25.nii.gz",
            "organ_name": "stomach",
            "qc_flag": "review_tiny_prediction",
            "qc_reason": "Smallest stomach prediction; likely undersegmented.",
            "use_for_volume_stats": False,
        },
        {
            "file_name": "case65_day28.nii.gz",
            "organ_name": "kidney_left",
            "qc_flag": "review_tiny_prediction",
            "qc_reason": "Smallest left kidney prediction; same scan has multiple tiny organs.",
            "use_for_volume_stats": False,
        },
        {
            "file_name": "case65_day28.nii.gz",
            "organ_name": "kidney_right",
            "qc_flag": "review_tiny_prediction",
            "qc_reason": "Smallest right kidney prediction; same scan has multiple tiny organs.",
            "use_for_volume_stats": False,
        },
        {
            "file_name": "case65_day28.nii.gz",
            "organ_name": "small_bowel",
            "qc_flag": "review_tiny_prediction",
            "qc_reason": "Smallest small bowel prediction; same scan has multiple tiny organs.",
            "use_for_volume_stats": False,
        },
        {
            "file_name": "case65_day0.nii.gz",
            "organ_name": "spleen",
            "qc_flag": "review_large_prediction",
            "qc_reason": "Largest spleen prediction; visually very large.",
            "use_for_volume_stats": True,
        },
        {
            "file_name": "case119_day21.nii.gz",
            "organ_name": "stomach",
            "qc_flag": "review_large_prediction",
            "qc_reason": "Largest stomach prediction.",
            "use_for_volume_stats": True,
        },
        {
            "file_name": "case119_day21.nii.gz",
            "organ_name": "small_bowel",
            "qc_flag": "review_large_prediction",
            "qc_reason": "Largest small bowel prediction.",
            "use_for_volume_stats": True,
        },
        {
            "file_name": "case134_day22.nii.gz",
            "organ_name": "colon",
            "qc_flag": "review_large_prediction",
            "qc_reason": "Largest colon prediction.",
            "use_for_volume_stats": True,
        },
        {
            "file_name": "case77_day20.nii.gz",
            "organ_name": "colon",
            "qc_flag": "review_small_prediction",
            "qc_reason": "Smallest colon prediction.",
            "use_for_volume_stats": True,
        },
    ]

    for flag in manual_flags:
        mask = (
            (qc_df["file_name"] == flag["file_name"])
            & (qc_df["organ_name"] == flag["organ_name"])
        )

        qc_df.loc[mask, "qc_flag"] = flag["qc_flag"]
        qc_df.loc[mask, "qc_reason"] = flag["qc_reason"]
        qc_df.loc[mask, "use_for_volume_stats"] = flag["use_for_volume_stats"]

    return qc_df


def apply_qc_to_organ_long_df(organ_long_df, qc_df):
    """
    Add QC decisions to organ_long_df.
    Create a cleaned volume column where excluded cases become NaN.
    """
    merged = organ_long_df.merge(
        qc_df[
            [
                "file_name",
                "organ_name",
                "qc_flag",
                "qc_reason",
                "use_for_volume_stats",
            ]
        ],
        on=["file_name", "organ_name"],
        how="left",
    )

    merged["qc_flag"] = merged["qc_flag"].fillna("ok")
    merged["qc_reason"] = merged["qc_reason"].fillna("")
    merged["use_for_volume_stats"] = merged["use_for_volume_stats"].fillna(True)

    merged["volume_ml_qc"] = merged["volume_ml_for_stats"]

    merged.loc[~merged["use_for_volume_stats"], "volume_ml_qc"] = np.nan

    return merged


def build_qc_filtered_volume_stats(organ_long_qc_df):
    """
    Build volume stats after QC exclusions.
    """
    stats_df = (
        organ_long_qc_df
        .groupby("organ_name")["volume_ml_qc"]
        .agg(
            n_used="count",
            mean_ml="mean",
            std_ml="std",
            min_ml="min",
            median_ml="median",
            max_ml="max",
        )
        .reset_index()
    )

    return stats_df.sort_values("organ_name").reset_index(drop=True)


def build_qc_flag_counts(qc_df):
    """
    Count QC flags per organ.
    """
    return (
        qc_df
        .groupby(["organ_name", "qc_flag"])["file_name"]
        .count()
        .reset_index(name="n")
        .sort_values(["organ_name", "qc_flag"])
        .reset_index(drop=True)
    )