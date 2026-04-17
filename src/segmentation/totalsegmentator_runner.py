from pathlib import Path
from typing import List, Optional, Union
import json

from totalsegmentator.python_api import totalsegmentator


def load_json(path: Union[str, Path]) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def case_id_from_path(path: Union[str, Path]) -> str:
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def list_nifti_cases(public_root: Union[str, Path]) -> List[Path]:
    public_root = Path(public_root)
    return sorted(public_root.glob("*.nii.gz"))


def run_totalsegmentator_on_case(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    task: str = "total_mr",
    device: str = "gpu",
    fast: bool = True,
    ml: bool = True,
    roi_subset: Optional[List[str]] = None,
    body_seg: bool = False,
    force_split: bool = False,
    statistics: bool = False,
    skip_existing: bool = True,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input scan not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists():
        print(f"[SKIP] Already exists: {output_path}")
        return output_path

    print(f"[RUN ] {input_path.name}")
    print(f"       task={task}, device={device}, fast={fast}, ml={ml}")

    totalsegmentator(
        input=input_path,
        output=output_path,
        ml=ml,
        fast=fast,
        task=task,
        roi_subset=roi_subset,
        body_seg=body_seg,
        force_split=force_split,
        statistics=statistics,
        device=device,
        quiet=False,
        verbose=False,
    )

    print(f"[DONE] Saved: {output_path}")
    return output_path


def run_totalsegmentator_on_folder(
    public_root: Union[str, Path],
    output_root: Union[str, Path],
    task: str = "total_mr",
    device: str = "gpu",
    fast: bool = True,
    ml: bool = True,
    roi_subset: Optional[List[str]] = None,
    body_seg: bool = False,
    force_split: bool = False,
    statistics: bool = False,
    skip_existing: bool = True,
) -> None:
    public_root = Path(public_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = list_nifti_cases(public_root)

    if len(cases) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in: {public_root}")

    print(f"Found {len(cases)} NIfTI files in {public_root}")

    for case_path in cases:
        case_id = case_id_from_path(case_path)

        if ml:
            output_path = output_root / f"{case_id}.nii.gz"
        else:
            output_path = output_root / case_id

        run_totalsegmentator_on_case(
            input_path=case_path,
            output_path=output_path,
            task=task,
            device=device,
            fast=fast,
            ml=ml,
            roi_subset=roi_subset,
            body_seg=body_seg,
            force_split=force_split,
            statistics=statistics,
            skip_existing=skip_existing,
        )