from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti
except Exception:
    load_multilabel_nifti = None


def _normalize_label_map(raw_label_map):
    id_to_name = {}

    if raw_label_map is None:
        return id_to_name

    for k, v in raw_label_map.items():
        try:
            if str(k).isdigit():
                id_to_name[int(k)] = str(v)
            elif str(v).isdigit():
                id_to_name[int(v)] = str(k)
        except Exception:
            continue

    return id_to_name


def load_multilabel_mask(mask_path):
    mask_path = Path(mask_path)

    if load_multilabel_nifti is not None:
        try:
            img, raw_label_map = load_multilabel_nifti(mask_path)
            mask = np.asarray(img.get_fdata(), dtype=np.int32)
            label_map = _normalize_label_map(raw_label_map)
            return mask, img, label_map
        except Exception:
            pass

    img = nib.load(str(mask_path))
    mask = np.asarray(img.get_fdata(), dtype=np.int32)
    return mask, img, {}


def get_spacing(img):
    zooms = img.header.get_zooms()
    return tuple(float(z) for z in zooms[:3])


def get_present_labels(mask):
    labels = np.unique(mask)
    labels = labels[labels > 0]
    return [int(x) for x in labels.tolist()]


def per_label_table(mask, label_map=None):
    if label_map is None:
        label_map = {}

    labels, counts = np.unique(mask, return_counts=True)

    rows = []
    total_voxels = int(mask.size)
    labeled_voxels = int((mask > 0).sum())

    for label_id, voxel_count in zip(labels, counts):
        if label_id == 0:
            continue

        label_id = int(label_id)
        voxel_count = int(voxel_count)

        rows.append(
            {
                "label_id": label_id,
                "label_name": label_map.get(label_id, f"label_{label_id}"),
                "voxel_count": voxel_count,
                "fraction_of_labeled_voxels": voxel_count / labeled_voxels if labeled_voxels > 0 else 0.0,
                "fraction_of_all_voxels": voxel_count / total_voxels if total_voxels > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "label_id",
                "label_name",
                "voxel_count",
                "fraction_of_labeled_voxels",
                "fraction_of_all_voxels",
            ]
        )

    return df.sort_values("voxel_count", ascending=False).reset_index(drop=True)


def count_nonempty_slices(mask):
    binary = mask > 0

    return {
        "slices_with_mask_axis0": int(np.sum(binary.sum(axis=(1, 2)) > 0)),
        "slices_with_mask_axis1": int(np.sum(binary.sum(axis=(0, 2)) > 0)),
        "slices_with_mask_axis2": int(np.sum(binary.sum(axis=(0, 1)) > 0)),
    }


def best_slice_from_mask(mask, axis=2):
    binary = mask > 0

    if axis == 0:
        scores = binary.sum(axis=(1, 2))
    elif axis == 1:
        scores = binary.sum(axis=(0, 2))
    elif axis == 2:
        scores = binary.sum(axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    return int(np.argmax(scores))


def get_bounding_box(mask):
    coords = np.argwhere(mask > 0)

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


def summarize_mask_file(mask_path):
    mask_path = Path(mask_path)

    mask, img, label_map = load_multilabel_mask(mask_path)
    spacing_x, spacing_y, spacing_z = get_spacing(img)

    nonzero_voxels = int((mask > 0).sum())
    total_voxels = int(mask.size)

    slice_counts = count_nonempty_slices(mask)
    bbox = get_bounding_box(mask)
    labels_present = get_present_labels(mask)

    summary = pd.DataFrame(
        [
            {
                "file_name": mask_path.name,
                "shape_x": int(mask.shape[0]),
                "shape_y": int(mask.shape[1]),
                "shape_z": int(mask.shape[2]),
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "spacing_z": spacing_z,
                "n_labels_present": int(len(labels_present)),
                "label_ids_present": ",".join(map(str, labels_present)),
                "nonzero_voxels": nonzero_voxels,
                "nonzero_fraction": nonzero_voxels / total_voxels if total_voxels > 0 else 0.0,
                "best_slice_axis2": best_slice_from_mask(mask, axis=2),
                **slice_counts,
                **bbox,
            }
        ]
    )

    labels_df = per_label_table(mask, label_map=label_map)

    return summary, labels_df, mask, img, label_map


def summarize_mask_folder(mask_root):
    mask_root = Path(mask_root)
    mask_paths = sorted(mask_root.glob("*.nii.gz"))

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in {mask_root}")

    all_summary = []
    all_labels = []

    for mask_path in mask_paths:
        summary_df, labels_df, _, _, _ = summarize_mask_file(mask_path)

        all_summary.append(summary_df)

        if len(labels_df) > 0:
            labels_df = labels_df.copy()
            labels_df["file_name"] = mask_path.name
            all_labels.append(labels_df)

    summary_out = pd.concat(all_summary, ignore_index=True)
    labels_out = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()

    return summary_out, labels_out, mask_paths


def plot_label_sizes_single_case(labels_df, top_n=15):
    if len(labels_df) == 0:
        print("No non-zero labels found.")
        return

    plot_df = labels_df.head(top_n).copy()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["label_name"], plot_df["voxel_count"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Voxel count")
    plt.title(f"Top {min(top_n, len(plot_df))} labels by voxel count")
    plt.tight_layout()
    plt.show()


def plot_dataset_nonzero_fraction(summary_df):
    if len(summary_df) == 0:
        print("No summary rows found.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(summary_df["nonzero_fraction"], bins=20)
    plt.xlabel("Nonzero fraction")
    plt.ylabel("Number of files")
    plt.title("Distribution of mask occupancy across files")
    plt.tight_layout()
    plt.show()


def plot_label_frequency_across_files(all_labels_df, top_n=15):
    if len(all_labels_df) == 0:
        print("No label rows found.")
        return

    freq_df = (
        all_labels_df.groupby("label_name")["file_name"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="n_files")
    )

    plt.figure(figsize=(10, 5))
    plt.bar(freq_df["label_name"], freq_df["n_files"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Number of files")
    plt.title(f"Top {min(top_n, len(freq_df))} labels by file frequency")
    plt.tight_layout()
    plt.show()


def plot_slice_occupancy(mask, axis=2):
    binary = mask > 0

    if axis == 0:
        scores = binary.sum(axis=(1, 2))
    elif axis == 1:
        scores = binary.sum(axis=(0, 2))
    elif axis == 2:
        scores = binary.sum(axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    plt.figure(figsize=(10, 4))
    plt.plot(scores)
    plt.xlabel("Slice index")
    plt.ylabel("Labeled voxels")
    plt.title(f"Mask occupancy per slice (axis={axis})")
    plt.tight_layout()
    plt.show()


def show_mask_slice(mask, slice_idx=None, axis=2):
    if slice_idx is None:
        slice_idx = best_slice_from_mask(mask, axis=axis)

    if axis == 0:
        mask_slice = mask[slice_idx, :, :]
    elif axis == 1:
        mask_slice = mask[:, slice_idx, :]
    elif axis == 2:
        mask_slice = mask[:, :, slice_idx]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    overlay = np.ma.masked_where(mask_slice <= 0, mask_slice)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay, cmap="jet", origin="lower")
    plt.title(f"Mask only | axis={axis}, slice={slice_idx}")
    plt.axis("off")
    plt.show()







GI_LABELS_TOTAL_MR = {
    6: "stomach",
    13: "small_bowel",
    15: "colon",
}


def filter_to_selected_labels(mask, selected_label_ids):
    """
    Keep only the selected labels.
    Everything else becomes 0.
    """
    selected_label_ids = set(int(x) for x in selected_label_ids)
    filtered = np.where(np.isin(mask, list(selected_label_ids)), mask, 0)
    return filtered.astype(np.int32)


def per_selected_label_table(mask, selected_labels):
    """
    selected_labels: dict like {6: "stomach", 13: "small_bowel", 15: "colon"}
    """
    rows = []
    total_voxels = int(mask.size)
    labeled_voxels = int((mask > 0).sum())

    for label_id, label_name in selected_labels.items():
        voxel_count = int((mask == label_id).sum())

        rows.append(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                "voxel_count": voxel_count,
                "present": voxel_count > 0,
                "fraction_of_selected_labeled_voxels": voxel_count / labeled_voxels if labeled_voxels > 0 else 0.0,
                "fraction_of_all_voxels": voxel_count / total_voxels if total_voxels > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("voxel_count", ascending=False).reset_index(drop=True)


def voxel_volume_mm3(img):
    spacing_x, spacing_y, spacing_z = get_spacing(img)
    return float(spacing_x * spacing_y * spacing_z)


def per_selected_label_table_with_volume(mask, img, selected_labels):
    df = per_selected_label_table(mask, selected_labels).copy()
    one_voxel_mm3 = voxel_volume_mm3(img)
    df["volume_mm3"] = df["voxel_count"] * one_voxel_mm3
    df["volume_ml"] = df["volume_mm3"] / 1000.0
    return df


def label_slice_span(mask, label_id, axis=2):
    binary = (mask == label_id)

    if axis == 0:
        scores = binary.sum(axis=(1, 2))
    elif axis == 1:
        scores = binary.sum(axis=(0, 2))
    elif axis == 2:
        scores = binary.sum(axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    present = np.where(scores > 0)[0]

    if len(present) == 0:
        return {
            "first_slice": None,
            "last_slice": None,
            "n_slices_present": 0,
            "best_slice": None,
        }

    best_slice = int(np.argmax(scores))

    return {
        "first_slice": int(present.min()),
        "last_slice": int(present.max()),
        "n_slices_present": int(len(present)),
        "best_slice": best_slice,
    }


def selected_label_span_table(mask, selected_labels, axis=2):
    rows = []

    for label_id, label_name in selected_labels.items():
        span = label_slice_span(mask, label_id, axis=axis)
        rows.append(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                **span,
            }
        )

    return pd.DataFrame(rows)


def summarize_selected_labels_file(mask_path, selected_labels=GI_LABELS_TOTAL_MR):
    """
    GI-focused summary for one mask file.
    """
    mask_path = Path(mask_path)

    mask, img, _ = load_multilabel_mask(mask_path)
    filtered_mask = filter_to_selected_labels(mask, selected_labels.keys())

    summary = pd.DataFrame(
        [
            {
                "file_name": mask_path.name,
                "shape_x": int(filtered_mask.shape[0]),
                "shape_y": int(filtered_mask.shape[1]),
                "shape_z": int(filtered_mask.shape[2]),
                "n_selected_labels_present": int(len(np.unique(filtered_mask[filtered_mask > 0]))),
                "selected_nonzero_voxels": int((filtered_mask > 0).sum()),
                "selected_nonzero_fraction": float((filtered_mask > 0).sum() / filtered_mask.size),
            }
        ]
    )

    labels_df = per_selected_label_table_with_volume(filtered_mask, img, selected_labels)
    spans_df = selected_label_span_table(filtered_mask, selected_labels, axis=2)

    return summary, labels_df, spans_df, filtered_mask, img


def summarize_selected_labels_folder(mask_root, selected_labels=GI_LABELS_TOTAL_MR):
    mask_root = Path(mask_root)
    mask_paths = sorted(mask_root.glob("*.nii.gz"))

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in {mask_root}")

    all_summary = []
    all_labels = []
    all_spans = []

    for mask_path in mask_paths:
        summary_df, labels_df, spans_df, _, _ = summarize_selected_labels_file(
            mask_path,
            selected_labels=selected_labels,
        )

        all_summary.append(summary_df)

        labels_df = labels_df.copy()
        labels_df["file_name"] = mask_path.name
        all_labels.append(labels_df)

        spans_df = spans_df.copy()
        spans_df["file_name"] = mask_path.name
        all_spans.append(spans_df)

    summary_out = pd.concat(all_summary, ignore_index=True)
    labels_out = pd.concat(all_labels, ignore_index=True)
    spans_out = pd.concat(all_spans, ignore_index=True)

    return summary_out, labels_out, spans_out, mask_paths


def plot_selected_label_sizes_single_case(labels_df):
    plot_df = labels_df.copy()

    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["label_name"], plot_df["voxel_count"])
    plt.ylabel("Voxel count")
    plt.title("GI labels by voxel count")
    plt.tight_layout()
    plt.show()


def plot_selected_label_volumes_single_case(labels_df):
    plot_df = labels_df.copy()

    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["label_name"], plot_df["volume_ml"])
    plt.ylabel("Volume (mL)")
    plt.title("GI labels by volume")
    plt.tight_layout()
    plt.show()


def plot_selected_label_frequency_across_files(all_labels_df):
    freq_df = (
        all_labels_df[all_labels_df["present"]]
        .groupby("label_name")["file_name"]
        .nunique()
        .reset_index(name="n_files")
    )

    plt.figure(figsize=(8, 4))
    plt.bar(freq_df["label_name"], freq_df["n_files"])
    plt.ylabel("Number of files")
    plt.title("GI label frequency across files")
    plt.tight_layout()
    plt.show()


def build_case_gi_summary_table(labels_df, spans_df, organ_order=("colon", "stomach", "small_bowel")):
    """
    Merge the single-case GI volume table and slice-span table into one easy table.
    Expects:
    - labels_df from summarize_selected_labels_file(...)
    - spans_df from summarize_selected_labels_file(...)
    """
    labels_small = labels_df[["label_id", "label_name", "voxel_count", "volume_ml"]].copy()
    spans_small = spans_df[["label_id", "label_name", "first_slice", "last_slice", "n_slices_present", "best_slice"]].copy()

    merged = labels_small.merge(
        spans_small,
        on=["label_id", "label_name"],
        how="left"
    )

    order_map = {name: i for i, name in enumerate(organ_order)}
    merged["sort_order"] = merged["label_name"].map(order_map).fillna(999)
    merged = merged.sort_values(["sort_order", "label_name"]).drop(columns="sort_order").reset_index(drop=True)

    return merged


def build_case_gi_summary_sentence(
    labels_df,
    spans_df,
    file_name=None,
    organ_order=("colon", "stomach", "small_bowel"),
):
    """
    Example output:
    case101_day20.nii.gz: colon 582.0 mL across 63 slices; stomach 578.4 mL across 35 slices; small_bowel 203.0 mL across 46 slices.
    """
    summary_table = build_case_gi_summary_table(labels_df, spans_df, organ_order=organ_order)

    parts = []
    for _, row in summary_table.iterrows():
        parts.append(
            f"{row['label_name']} {row['volume_ml']:.1f} mL across {int(row['n_slices_present'])} slices"
        )

    prefix = f"{file_name}: " if file_name else ""
    return prefix + "; ".join(parts) + "."


def get_label_id_by_name(selected_labels, label_name):
    """
    selected_labels example:
    {6: 'stomach', 13: 'small_bowel', 15: 'colon'}
    """
    for label_id, name in selected_labels.items():
        if name == label_name:
            return int(label_id)
    raise KeyError(f"Label name not found: {label_name}")


def extract_single_label_mask(mask, label_id):
    """
    Keep only one label in the mask.
    Everything else becomes 0.
    """
    label_id = int(label_id)
    return np.where(mask == label_id, mask, 0).astype(np.int32)


def build_case_gi_summary_table(labels_df, spans_df, organ_order=("colon", "stomach", "small_bowel")):
    labels_small = labels_df[["label_id", "label_name", "voxel_count", "volume_ml"]].copy()
    spans_small = spans_df[["label_id", "label_name", "first_slice", "last_slice", "n_slices_present", "best_slice"]].copy()

    merged = labels_small.merge(
        spans_small,
        on=["label_id", "label_name"],
        how="left"
    )

    order_map = {name: i for i, name in enumerate(organ_order)}
    merged["sort_order"] = merged["label_name"].map(order_map).fillna(999)
    merged = merged.sort_values(["sort_order", "label_name"]).drop(columns="sort_order").reset_index(drop=True)

    return merged


def build_case_gi_summary_sentence(
    labels_df,
    spans_df,
    file_name=None,
    organ_order=("colon", "stomach", "small_bowel"),
):
    summary_table = build_case_gi_summary_table(labels_df, spans_df, organ_order=organ_order)

    parts = []
    for _, row in summary_table.iterrows():
        parts.append(
            f"{row['label_name']} {row['volume_ml']:.1f} mL across {int(row['n_slices_present'])} slices"
        )

    prefix = f"{file_name}: " if file_name else ""
    return prefix + "; ".join(parts) + "."


def get_label_id_by_name(selected_labels, label_name):
    for label_id, name in selected_labels.items():
        if name == label_name:
            return int(label_id)
    raise KeyError(f"Label name not found: {label_name}")


def extract_single_label_mask(mask, label_id):
    label_id = int(label_id)
    return np.where(mask == label_id, mask, 0).astype(np.int32)