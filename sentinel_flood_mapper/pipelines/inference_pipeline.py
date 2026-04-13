"""
inference_pipeline.py

Inference pipeline for flood extent prediction from raw Sentinel-1 SAR imagery.

Accepts either a single chip or a directory of chips. Handles preprocessing,
inference and saving georeferenced output GeoTIFFs. Optionally computes
metrics and generates contingency maps if ground truth labels are provided.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
import torch
from tqdm import tqdm
from sentinel_flood_mapper import Config
from sentinel_flood_mapper.data import normalise_sen1floods11, remap_labels
from sentinel_flood_mapper.models import get_model, predict, SegmentationMetrics
from sentinel_flood_mapper.utils import (
    load_checkpoint,
    plot_contingency_map,
    plot_predictions
)

class InferencePipeline:
    """
    Inference pipeline for generating flood extent predictions from raw
    Sentinel-1 SAR imagery.

    Accepts a single chip or a directory of chips. Preprocesses raw SAR
    images using the same normalisation pipeline as training, runs inference
    and saves georeferenced probability and binary mask GeoTIFFs.

    Optionally computes metrics and generates contingency maps if ground
    truth labels are provided.
    """
    def __init__(
            self,
            config: Config,
            checkpoint_path: Optional[Path] = None,
            image_path: Optional[Path] = None,
            image_dir: Optional[Path] = None,
            label_path: Optional[Path] = None,
            label_dir: Optional[Path] = None,
            output_dir: Optional[Path] = None,
            use_tiling: bool = False,
            stride: int = 64,
    ) -> None:
        """
        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint. If None uses
                config.paths.checkpoints_dir / 'best_model.pt'
            image_path: Path to a single raw SAR chip GeoTIFF
            image_dir: Path to a directory of raw SAR chip GeoTIFFs
            label_path: Path to a single ground truth label GeoTIFF
            label_dir: Path to a directory of ground truth label GeoTIFFs
            output_dir: Directory to save prediction outputs. If None uses
                config.paths.figures / 'inference'
            use_tiling: If True, split image into 128x128 tiles for inference.
                If False, run as single forward pass with padding. Default False
            stride: Step size between tiles in pixels. Default 64 (50% overlap).

        Raises:
            ValueError: If neither or both of image_path and image_dir are
                provided
        """
        # Validate image input
        if image_path is None and image_dir is None:
            raise ValueError(
                "Either image_path or image_dir must be provided"
            )
        if image_path is not None and image_dir is not None:
            raise ValueError(
                "Only one of image_path or image_dir may be provided, not both"
            )

        self.config = config
        self.use_tiling = use_tiling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(
            checkpoint_path or config.paths.checkpoints_dir / 'best_model.pt'
        )
        self.output_dir = Path(
            output_dir or config.paths.figures / 'inference'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stride = stride

        # Resolve image and label pairs
        if image_path is not None:
            self.image_paths = [Path(image_path)]
            self.label_paths = [Path(label_path)] if label_path else []
        else:
            self.image_paths = sorted(Path(image_dir).glob("*.tif"))
            if label_dir is not None:
                self.label_paths = sorted(Path(label_dir).glob("*.tif"))
            else:
                self.label_paths = []

        # Populated by load_model
        self.model = None

        # Results accumulated across all chips
        self.all_results = {}

        print(f"Using device: {self.device}")
        print(f"Checkpoint: {self.checkpoint_path.relative_to(self.config.paths.project_root)}")
        print(f"Number of images to infer: {len(self.image_paths)}")
        print(f"Output dir: {self.output_dir.relative_to(self.config.paths.project_root)}")

    def load_model(
            self
    ) -> None:
        """
        Build model architecture and load weights from checkpoint.
        """
        print("\n" + "=" * 70)
        print("LOADING MODEL")
        print("=" * 70)

        self.model, _ = load_checkpoint(
            model=get_model(
                model_config=self.config.model,
                device=self.device,
            ),
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()

    def _preprocess(
            self,
            s1_arr: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Sen1Floods11 preprocessing to a raw SAR array.

        Args:
            s1_arr: Raw SAR array of shape (2, H, W) in dB scale

        Returns:
            Normalised array of shape (2, H, W), values in [0, 1]
        """
        pre = self.config.preprocessing.sen1floods11
        return normalise_sen1floods11(
            arr=s1_arr,
            nan_fill=pre.nan_fill,
            clip_min=pre.clip_min,
            clip_max=pre.clip_max,
            apply_lee_filter=pre.apply_lee_filter,
            window_size=pre.lee_window_size,
        )

    def _save_prediction(
            self,
            prob_map: np.ndarray,
            binary_mask: np.ndarray,
            src_meta: dict,
            chip_name: str,
    ) -> tuple[Path, Path]:
        """
        Save probability map and binary mask as georeferenced GeoTIFFs.

        Args:
            prob_map: Float32 probability array of shape (H, W)
            binary_mask: Uint8 binary mask array of shape (H, W)
            src_meta: Rasterio metadata from the source SAR chip
            chip_name: Base name for output files

        Returns:
            Tuple of (prob_path, binary_path) output file paths
        """
        prob_path = self.output_dir / f"{chip_name}_prediction_prob.tif"
        binary_path = self.output_dir / f"{chip_name}_prediction_binary.tif"

        # Probability map metadata
        prob_meta = {
            **src_meta,
            "count": 1,
            "dtype": "float32",
            "nodata": None,
        }
        with rasterio.open(prob_path, "w", **prob_meta) as dst:
            dst.write(prob_map[np.newaxis, :, :])

        # Binary mask metadata
        binary_meta = {
            **src_meta,
            "count": 1,
            "dtype": "uint8",
            "nodata": 255,
        }
        with rasterio.open(binary_path, "w", **binary_meta) as dst:
            dst.write(binary_mask[np.newaxis, :, :])

        return prob_path, binary_path

    def _process_chip(
            self,
            image_path: Path,
            label_path: Optional[Path] = None,
    ) -> dict:
        """
        Run the full inference pipeline on a single SAR chip.

        Args:
            image_path: Path to raw SAR chip GeoTIFF
            label_path: Optional path to ground truth label GeoTIFF

        Returns:
            Dictionary containing prediction paths and optionally metrics
        """
        chip_name = image_path.stem.replace("_S1Hand", "")
        print(f"\nProcessing {chip_name}...")

        # Load raw SAR image
        with rasterio.open(image_path) as src:
            s1_arr = src.read().astype(np.float32)
            src_meta = src.meta.copy()

        # Preprocess
        s1_norm = self._preprocess(s1_arr)

        # Run inference
        prob_map, binary_mask = predict(
            model=self.model,
            arr=s1_norm,
            device=self.device,
            use_tiling=self.use_tiling,
            threshold=self.config.inference.threshold,
            stride=self.stride,
        )

        # Save prediction GeoTIFFs
        prob_path, binary_path = self._save_prediction(
            prob_map=prob_map,
            binary_mask=binary_mask,
            src_meta=src_meta,
            chip_name=chip_name,
        )

        result = {
            "chip_name": chip_name,
            "prob_path": prob_path,
            "binary_path": binary_path,
        }

        # If ground truth provided - compute metrics and save visualisations
        if label_path is not None:
            with rasterio.open(label_path) as src:
                label_arr = src.read(1).astype(np.int64)

            # Remap labels using Sen1Floods11 label map
            label_remapped = remap_labels(
                arr=label_arr,
                label_map=self.config.label_maps.sen1floods11,
            )

            # Create nodata mask from og raw array
            sar_nodata_mask = np.isnan(s1_arr).any(axis=0)      # (H, W)

            # Set label to ignore index where SAR is nodata
            label_remapped[sar_nodata_mask] = 255

            # Compute metrics
            metrics = SegmentationMetrics(device=torch.device("cpu"))
            pred_tensor = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
            label_tensor = torch.from_numpy(label_remapped).unsqueeze(0)
            metrics.update(pred_tensor, label_tensor, is_probability=True)
            chip_metrics = metrics.compute()

            print(f"IoU: {chip_metrics['iou']:.4f}")
            print(f"F1: {chip_metrics['f1']:.4f}")
            print(f"Precision: {chip_metrics['precision']:.4f}")
            print(f"Recall: {chip_metrics['recall']:.4f}")
            print(f"Accuracy: {chip_metrics['accuracy']:.4f}")

            # Save contingency map
            plot_contingency_map(
                image=s1_norm,
                label=label_remapped,
                prediction=prob_map,
                threshold=self.config.inference.threshold,
                save_path=self.output_dir / f"{chip_name}_contingency.png",
                show=False,
            )

            result["metrics"] = chip_metrics
        return result

    def run_inference(
            self
    ) -> dict:
        """
        Run inference on all chips.

        Returns:
            Dictionary mapping chip names to their result dictionaries
        """
        print("\n" + "=" * 70)
        print("RUNNING INFERENCE")
        print("=" * 70)

        # Build label path lookup if label paths provided
        label_lookup = {}
        if self.label_paths:
            for lp in self.label_paths:
                label_lookup[lp.stem.replace("_LabelHand", "")] = lp

        for image_path in tqdm(self.image_paths, desc="Running inference", unit="chip"):
            chip_name = image_path.stem.replace("_S1Hand", "")
            label_path = label_lookup.get(chip_name)
            result = self._process_chip(image_path=image_path, label_path=label_path)
            self.all_results[chip_name] = result

        print(f"\nInference complete. {len(self.all_results)} chips processed.")
        print(f"Outputs saved to: {self.output_dir.relative_to(self.config.paths.project_root)}")

        return self.all_results

    def get_summary(
            self
    ) -> None:
        """
        Print a summary of the inference results.
        """
        print("\n" + "=" * 70)
        print("INFERENCE SUMMARY")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Chips processed: {len(self.all_results)}")
        print(f"Output directory: {self.output_dir.relative_to(self.config.paths.project_root)}")

        # If metrics available, print mean across all chips
        chips_with_metrics = [
            r for r in self.all_results.values() if "metrics" in r
        ]

        if chips_with_metrics:
            mean_iou = np.mean(
                [r["metrics"]["iou"] for r in chips_with_metrics]
            )
            mean_f1 = np.mean(
                [r["metrics"]["f1"] for r in chips_with_metrics]
            )
            print(f"\n Mean metrics across {len(chips_with_metrics)} chips:")
            print(f"IoU: {mean_iou:.4f}")
            print(f"F1: {mean_f1:.4f}")

        print("=" * 70)


def run_inference_pipeline(
        config: Config,
        checkpoint_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        label_path: Optional[Path] = None,
        label_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        use_tiling: bool = False,
        stride: int = 64,
) -> InferencePipeline:
    """
    Run the complete inference pipeline in one function call.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        image_path: Path to a single SAR chip
        image_dir: Path to a directory of SAR chips
        label_path: Path to a single ground truth label
        label_dir: Path to a directory of ground truth labels
        output_dir: Directory to save outputs
        use_tiling: Whether to use tiled inference. Default False
        stride: Step size between tiles in pixels. Default 64 (50% overlap).

    Returns:
        InferencePipeline instance after completion
    """
    pipeline = InferencePipeline(
        config=config,
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        image_dir=image_dir,
        label_path=label_path,
        label_dir=label_dir,
        output_dir=output_dir,
        use_tiling=use_tiling,
        stride=stride,
    )
    pipeline.load_model()
    pipeline.run_inference()
    pipeline.get_summary()

    return pipeline



