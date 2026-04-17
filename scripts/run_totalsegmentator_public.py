from pathlib import Path
import sys
import json
from multiprocessing import freeze_support

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.segmentation.totalsegmentator_runner import (
    run_totalsegmentator_on_case,
    run_totalsegmentator_on_folder,
)


def main():
    config_path = project_root / "configs" / "datasets" / "public_dataset.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    public_root = Path(config["public_root"])
    output_root = Path(config["totalseg_root"])

    # First test on ONE case only.
    # After this works, set single_case_name = None to run all files.
    single_case_name = None #"case84_day21.nii.gz"

    if single_case_name is not None:
        input_path = public_root / single_case_name

        if config.get("totalseg_ml", True):
            output_path = output_root / single_case_name
        else:
            output_path = output_root / single_case_name.replace(".nii.gz", "")

        print("Running single case:")
        print("input :", input_path)
        print("output:", output_path)

        run_totalsegmentator_on_case(
            input_path=input_path,
            output_path=output_path,
            task=config.get("totalseg_task", "total_mr"),
            device=config.get("totalseg_device", "cpu"),
            fast=config.get("totalseg_fast", True),
            ml=config.get("totalseg_ml", True),
            roi_subset=config.get("totalseg_roi_subset", None),
            body_seg=config.get("totalseg_body_seg", False),
            force_split=config.get("totalseg_force_split", False),
            statistics=config.get("totalseg_statistics", False),
            skip_existing=config.get("totalseg_skip_existing", True),
        )
    else:
        run_totalsegmentator_on_folder(
            public_root=public_root,
            output_root=output_root,
            task=config.get("totalseg_task", "total_mr"),
            device=config.get("totalseg_device", "cpu"),
            fast=config.get("totalseg_fast", True),
            ml=config.get("totalseg_ml", True),
            roi_subset=config.get("totalseg_roi_subset", None),
            body_seg=config.get("totalseg_body_seg", False),
            force_split=config.get("totalseg_force_split", False),
            statistics=config.get("totalseg_statistics", False),
            skip_existing=config.get("totalseg_skip_existing", True),
        )


if __name__ == "__main__":
    freeze_support()
    main()