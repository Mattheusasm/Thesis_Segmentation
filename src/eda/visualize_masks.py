from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_nifti(path):
    """
    Load a NIfTI file and return:
    - data: numpy array
    - img: nibabel image object
    """
    path = Path(path)
    img = nib.load(str(path))
    data = img.get_fdata()

    # if a volume is accidentally 4D with one channel/timepoint, squeeze it
    if data.ndim == 4 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)

    return data, img


def ensure_3d(volume, name="volume"):
    if volume.ndim != 3:
        raise ValueError(f"{name} must be 3D, but got shape {volume.shape}")


def normalize_slice(slice_2d):
    """
    Simple min-max normalization for display only.
    """
    slice_2d = np.asarray(slice_2d, dtype=np.float32)

    vmin = np.nanmin(slice_2d)
    vmax = np.nanmax(slice_2d)

    if vmax <= vmin:
        return np.zeros_like(slice_2d, dtype=np.float32)

    return (slice_2d - vmin) / (vmax - vmin)


def get_slice(volume, slice_idx, axis=2):
    """
    Extract one 2D slice from a 3D volume.
    axis=0,1,2
    """
    ensure_3d(volume, "volume")

    if axis == 0:
        return volume[slice_idx, :, :]
    if axis == 1:
        return volume[:, slice_idx, :]
    if axis == 2:
        return volume[:, :, slice_idx]

    raise ValueError("axis must be 0, 1, or 2")


def best_slice_from_mask(mask_volume, axis=2):
    """
    Pick the slice with the most mask pixels.
    Very useful so you do not land on an empty slice.
    """
    ensure_3d(mask_volume, "mask_volume")

    mask_binary = mask_volume > 0

    if axis == 0:
        scores = mask_binary.sum(axis=(1, 2))
    elif axis == 1:
        scores = mask_binary.sum(axis=(0, 2))
    elif axis == 2:
        scores = mask_binary.sum(axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    return int(np.argmax(scores))


def make_overlay_mask(mask_slice):
    """
    Convert mask slice to masked array so background is transparent.
    """
    mask_slice = np.asarray(mask_slice)
    return np.ma.masked_where(mask_slice <= 0, mask_slice)


def show_nifti_slice_with_mask(
    scan_volume,
    mask_volume,
    slice_idx=None,
    axis=2,
    alpha=0.45,
    title=None,
    save_path=None,
):
    """
    Overlay one mask slice on one scan slice.
    scan_volume and mask_volume must already be aligned and have same shape.
    """
    ensure_3d(scan_volume, "scan_volume")
    ensure_3d(mask_volume, "mask_volume")

    if scan_volume.shape != mask_volume.shape:
        raise ValueError(
            f"scan and mask must have same shape, got {scan_volume.shape} vs {mask_volume.shape}"
        )

    if slice_idx is None:
        slice_idx = best_slice_from_mask(mask_volume, axis=axis)

    scan_slice = get_slice(scan_volume, slice_idx, axis=axis)
    mask_slice = get_slice(mask_volume, slice_idx, axis=axis)

    scan_slice = normalize_slice(scan_slice)
    overlay = make_overlay_mask(mask_slice)

    plt.figure(figsize=(6, 6))
    plt.imshow(scan_slice, cmap="gray", origin="lower")
    plt.imshow(overlay, cmap="jet", alpha=alpha, origin="lower")
    plt.axis("off")

    if title is None:
        title = f"axis={axis}, slice={slice_idx}"
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def show_nifti_slice_only(
    scan_volume,
    slice_idx,
    axis=2,
    title=None,
    save_path=None,
):
    """
    Show scan only, without mask.
    """
    ensure_3d(scan_volume, "scan_volume")

    scan_slice = get_slice(scan_volume, slice_idx, axis=axis)
    scan_slice = normalize_slice(scan_slice)

    plt.figure(figsize=(6, 6))
    plt.imshow(scan_slice, cmap="gray", origin="lower")
    plt.axis("off")

    if title is None:
        title = f"axis={axis}, slice={slice_idx}"
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


    from pathlib import Path
import nibabel as nib


def load_nifti(path):
    path = Path(path)
    img = nib.load(str(path))
    data = img.get_fdata()

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    return data, img


def _normalize_for_display(slice_2d):
    slice_2d = slice_2d.astype("float32")
    vmin = slice_2d.min()
    vmax = slice_2d.max()

    if vmax <= vmin:
        return np.zeros_like(slice_2d, dtype="float32")

    return (slice_2d - vmin) / (vmax - vmin)


def _get_slice(volume, slice_idx, axis=2):
    if axis == 0:
        return volume[slice_idx, :, :]
    if axis == 1:
        return volume[:, slice_idx, :]
    if axis == 2:
        return volume[:, :, slice_idx]
    raise ValueError("axis must be 0, 1, or 2")


def best_slice_from_mask(mask_volume, axis=2):
    mask_binary = mask_volume > 0

    if axis == 0:
        scores = mask_binary.sum(axis=(1, 2))
    elif axis == 1:
        scores = mask_binary.sum(axis=(0, 2))
    elif axis == 2:
        scores = mask_binary.sum(axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    return int(np.argmax(scores))


def show_nifti_overlay_from_paths(
    scan_path,
    mask_path,
    axis=2,
    slice_idx=None,
    alpha=0.45,
    title=None,
    save_path=None,
):
    scan_volume, _ = load_nifti(scan_path)
    mask_volume, _ = load_nifti(mask_path)

    if scan_volume.shape != mask_volume.shape:
        raise ValueError(
            f"Shape mismatch: scan {scan_volume.shape} vs mask {mask_volume.shape}"
        )

    if slice_idx is None:
        slice_idx = best_slice_from_mask(mask_volume, axis=axis)

    scan_slice = _get_slice(scan_volume, slice_idx, axis=axis)
    mask_slice = _get_slice(mask_volume, slice_idx, axis=axis)

    scan_slice = _normalize_for_display(scan_slice)
    mask_overlay = np.ma.masked_where(mask_slice <= 0, mask_slice)

    plt.figure(figsize=(6, 6))
    plt.imshow(scan_slice, cmap="gray", origin="lower")
    plt.imshow(mask_overlay, cmap="jet", alpha=alpha, origin="lower")
    plt.axis("off")

    if title is None:
        title = f"slice={slice_idx}, axis={axis}"
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def show_nifti_overlay_from_arrays(
    scan_volume,
    mask_volume,
    axis=2,
    slice_idx=None,
    alpha=0.45,
    title=None,
    save_path=None,
):
    if scan_volume.shape != mask_volume.shape:
        raise ValueError(
            f"Shape mismatch: scan {scan_volume.shape} vs mask {mask_volume.shape}"
        )

    if slice_idx is None:
        slice_idx = best_slice_from_mask(mask_volume, axis=axis)

    scan_slice = _get_slice(scan_volume, slice_idx, axis=axis)
    mask_slice = _get_slice(mask_volume, slice_idx, axis=axis)

    scan_slice = _normalize_for_display(scan_slice)
    mask_overlay = np.ma.masked_where(mask_slice <= 0, mask_slice)

    plt.figure(figsize=(6, 6))
    plt.imshow(scan_slice, cmap="gray", origin="lower")
    plt.imshow(mask_overlay, cmap="jet", alpha=alpha, origin="lower")
    plt.axis("off")

    if title is None:
        title = f"slice={slice_idx}, axis={axis}"
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()

def show_nifti_overlay_from_arrays(
    scan_volume,
    mask_volume,
    axis=2,
    slice_idx=None,
    alpha=0.45,
    title=None,
    save_path=None,
):
    if scan_volume.shape != mask_volume.shape:
        raise ValueError(
            f"Shape mismatch: scan {scan_volume.shape} vs mask {mask_volume.shape}"
        )

    if slice_idx is None:
        slice_idx = best_slice_from_mask(mask_volume, axis=axis)

    scan_slice = _get_slice(scan_volume, slice_idx, axis=axis)
    mask_slice = _get_slice(mask_volume, slice_idx, axis=axis)

    scan_slice = _normalize_for_display(scan_slice)
    mask_overlay = np.ma.masked_where(mask_slice <= 0, mask_slice)

    plt.figure(figsize=(6, 6))
    plt.imshow(scan_slice, cmap="gray", origin="lower")
    plt.imshow(mask_overlay, cmap="jet", alpha=alpha, origin="lower")
    plt.axis("off")

    if title is None:
        title = f"slice={slice_idx}, axis={axis}"
    plt.title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()