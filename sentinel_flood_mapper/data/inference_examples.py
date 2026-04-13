"""
inference_examples.py

Functions for preparing Bolivia inference example chips for use in the
predict pipeline demonstration.
"""
import shutil
from pathlib import Path

def save_inference_examples(
        s1_dir: Path,
        label_dir: Path,
        output_dir: Path,
        event: str = "Bolivia"
) -> None:
    """
    Copy raw unprocessed SAR chips and labels for a specified event to
    the inference examples directory.

    Files are saved in their original form - no normalisation, remapping or
    tiling is applied. The predict pipeline handles all preprocessing at inference time.

    Skips if output directory already exists and contains files.

    Args:
        s1_dir: Directory containing S1Hand GeoTIFF files
        label_dir: Directory containing LabelHand GeoTIFF files
        output_dir: Root directory to save inference example files.
                Subdirectories S1Hand/ and LabelHand/ are created inside
        event: Event name prefix to filter chips by. Default is 'Bolivia'

    Raises:
        FileNotFoundError: If no matching S1Hand files are found for the
                specified event in s1_dir
    """
    s1_out = Path(output_dir) / "S1Hand"
    label_out = Path(output_dir) / "LabelHand"

    # Skip if already saved
    if s1_out.exists() and any(s1_out.iterdir()):
        print(f"Inference examples already exist in {output_dir}.")
        print("Skipping.")
        return

    s1_files = sorted(Path(s1_dir).glob(f"{event}_*_S1Hand.tif"))

    if len(s1_files) == 0:
        raise FileNotFoundError(
            f"No S1Hand files found for event '{event}' in {s1_dir}."
        )

    s1_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(s1_files)} {event} chips to {output_dir}...")

    skipped = 0

    for s1_path in s1_files:
        chip_id = s1_path.name.replace("_S1Hand.tif", "")
        label_path = Path(label_dir) / f"{chip_id}_LabelHand.tif"

        if not label_path.exists():
            print(f"Warning: no label found for {chip_id}, skipping.")
            skipped += 1
            continue

        shutil.copy2(s1_path, s1_out / s1_path.name)
        shutil.copy2(label_path, label_out / label_path.name)

    saved = len(s1_files) - skipped
    print(f"Done. {saved} inference example chips saved to {output_dir}.")
    if skipped > 0:
        print(f"Warning: {skipped} chips skipped due to missing labels.")
