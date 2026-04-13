"""
training_pipeline.py

Training pipeline for the sentinel flood mapper project.
Handles preprocessing, dataloader creation, model setup and training.
"""
from typing import Optional
import torch
from sentinel_flood_mapper import Config
from sentinel_flood_mapper.data import (
    preprocess_sen1floods11,
    preprocess_sturm,
    save_inference_examples,
    build_file_pairs,
    create_splits,
    get_dataloader,
    Compose,
    RandomHFlip,
    RandomVFlip,
    RandomRotation90,
    RandomBrightnessContrast
)
from sentinel_flood_mapper.models import get_model, train
from sentinel_flood_mapper.utils import plot_training_curves

def _build_transforms(tr) -> Optional[Compose]:
    """
    Build training transforms from config.

    Args:
        tr: TransformConfig object.

    Returns:
        Compose object if any transforms are enabled, None otherwise
    """
    transform_map = [
        (tr.random_hflip, RandomHFlip, {}),
        (tr.random_vflip, RandomVFlip, {}),
        (tr.random_rotation90, RandomRotation90, {}),
        (tr.random_brightness_contrast, RandomBrightnessContrast, {
            "brightness_limit": tr.random_brightness_contrast.brightness_limit,
            "contrast_limit": tr.random_brightness_contrast.contrast_limit
        })
    ]

    transform_list = [
        cls(p=cfg.p, **kwargs)
        for cfg, cls, kwargs in transform_map
        if cfg.enabled
    ]

    return Compose(transform_list) if transform_list else None

class TrainingPipeline:
    def __init__(
            self,
            config: Config
    ) -> None:
        """
        Create TrainingPipeline object.

        Args:
            config: Configuration object containing all pipeline settings.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # data attributes - populated by run_preprocessing
        self.train_pairs = None
        self.val_pairs = None

        # dataloader attributes - populated by create_dataloaders
        self.train_loader = None
        self.val_loader = None

        # model attributes - populated by setup_model
        self.model = None

        # results attributes - populated by run_training
        self.history = None
        self.checkpoint_dir = None

    def run_preprocessing(self) -> None:
        """
        Run complete preprocessing pipeline.

        Steps:
            1. Preprocess Sen1Floods11 chips into normalised 128x128 tiles
            2. Preprocess STURM floodmap labels
            3. Save Bolivia inference examples
            4. Build train/val file pair lists from event splits
        """
        print("=" * 70)
        print("PREPROCESSING PIPELINE")
        print("=" * 70)

        paths = self.config.paths
        preprocessing = self.config.preprocessing
        label_maps = self.config.label_maps

        # --- Step 1: Preprocess Sen1Floods11 ---
        print("\n1. Preprocessing Sen1Floods11...")
        preprocess_sen1floods11(
            s1_dir=paths.raw_dir / "Sen1Floods11" / "S1Hand",
            label_dir=paths.raw_dir / "Sen1Floods11" / "LabelHand",
            output_dir=paths.processed_dir / "Sen1Floods11",
            nan_fill=preprocessing.sen1floods11.nan_fill,
            clip_min=preprocessing.sen1floods11.clip_min,
            clip_max=preprocessing.sen1floods11.clip_max,
            apply_lee_filter=preprocessing.sen1floods11.apply_lee_filter,
            lee_window_size=preprocessing.sen1floods11.lee_window_size,
            label_map=label_maps.sen1floods11
        )

        # --- Step 2: Preprocess STURM ---
        print("\n2. Preprocessing STURM...")
        preprocess_sturm(
            s1_dir=paths.raw_dir / "STURM" / "Sentinel1" / "S1",
            label_dir=paths.raw_dir / "STURM" / "Sentinel1" / "Floodmaps",
            output_dir=paths.processed_dir / "STURM",
            label_map=label_maps.sturm
        )

        # --- Step 3: Save inference examples ---
        print("\n3. Saving Bolivia inference examples...")
        save_inference_examples(
            s1_dir=paths.raw_dir / "Sen1Floods11" / "S1Hand",
            label_dir=paths.raw_dir / "Sen1Floods11" / "LabelHand",
            output_dir=paths.inference_examples_dir,
            event="Bolivia"
        )

        # --- Step 4: Create splits and build file pair lists ---
        print("\n4. Creating splits and building file pair lists...")

        if not paths.splits_file.exists():
            create_splits(config=self.config)
        else:
            print(f"Splits file already exists at {paths.splits_file}. Skipping.")

        self.train_pairs = build_file_pairs(
            splits_csv=paths.splits_file,
            split="train",
            project_root=paths.project_root
        )
        self.val_pairs = build_file_pairs(
            splits_csv=paths.splits_file,
            split="val",
            project_root=paths.project_root
        )

        print("\nPreprocessing complete.")
        print(f"Train pairs: {len(self.train_pairs)}")
        print(f"Val pairs: {len(self.val_pairs)}")

    def create_dataloaders(self) -> None:
        """
        Create train and validation dataloaders.

        Builds augmentation transforms for training.
        """
        print("\n" + "=" * 70)
        print("CREATING DATALOADERS")
        print("=" * 70)

        # Build training transforms from config
        train_transform = _build_transforms(self.config.transforms)

        self.train_loader = get_dataloader(
            file_pairs=self.train_pairs,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            transform=train_transform
        )
        self.val_loader = get_dataloader(
            file_pairs=self.val_pairs,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
        )

        print(f"Train loader: {len(self.train_loader.dataset):,} tiles, "
              f"{len(self.train_loader)} batches")
        print(f"Val loader: {len(self.val_loader.dataset):,} tiles, "
              f"{len(self.val_loader)} batches")

    def setup_model(self) -> None:
        """
        Build and initialise the U-Net model.
        """
        print("\n" + "=" * 70)
        print("SETTING UP MODEL")
        print("=" * 70)

        self.model = get_model(
            model_config=self.config.model,
            device=self.device
        )

    def run_training(self) -> None:
        """
        Run the training loop.

        Trains the model with validation, early stopping and checkpoint saving.
        Saves training curves to outputs/figures/
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        self.checkpoint_dir = self.config.paths.checkpoints_dir

        self.history = train(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config.train,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir
        )

        # Save training curves
        plot_training_curves(
            history=self.history,
            save_path=self.config.paths.figures / "training_curves.png",
            show=False
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Checkpoints save to {self.checkpoint_dir.relative_to(self.config.paths.project_root)}")

    def get_summary(self) -> None:
        """
        Print a summary of the current pipeline state.
        """
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Device: {self.device}")

        if self.train_pairs:
            print("\nFile pairs:")
            print(f"Train: {len(self.train_pairs)}")
            print(f"Val: {len(self.val_pairs)}")

        if self.train_loader:
            print("\nDataloaders:")
            print(f"Train: {len(self.train_loader.dataset)} tiles, "
                  f"{len(self.train_loader)} batches")
            print(f"Val: {len(self.val_loader.dataset)} tiles, "
                  f"{len(self.val_loader)} batches")

        if self.history:
            best_iou = max(self.history["val_iou"])
            best_f1 = max(self.history["val_f1"])
            print("\nTraining:")
            print(f"Epochs completed: {len(self.history['epoch'])}")
            print(f"Best val IoU: {best_iou:.4f}")
            print(f"Best val F1: {best_f1:.4f}")

        if self.checkpoint_dir:
            print(f"\nCheckpoint directory: {self.checkpoint_dir.relative_to(self.config.paths.project_root)}")

        print("=" * 70)

def run_training_pipeline(
        config: Config
) -> TrainingPipeline:
    """
    Run the complete training pipeline in one function call.

    Args:
        config: Configuration object

    Returns:
        TrainingPipeline instance after completion
    """
    pipeline = TrainingPipeline(config=config)
    pipeline.run_preprocessing()
    pipeline.create_dataloaders()
    pipeline.setup_model()
    pipeline.run_training()
    pipeline.get_summary()
    return pipeline











