"""
config.py

Loads default.yaml and creates a configuration object.
"""
from dataclasses import dataclass
from pathlib import Path
import yaml

# -- Dataclasses (1 per YAML section) --
@dataclass
class PathsConfig:
    """File and directory paths. All resolved to absolute paths at load time."""
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    checkpoints_dir: Path
    inference_checkpoint: Path
    predictions: Path
    figures: Path
    logs: Path
    splits_file: Path
    inference_examples_dir: Path

@dataclass
class Sen1Floods11PreprocessConfig:
    """Preprocessing parameters for Sen1Floods11 SAR imagery"""
    nan_fill: float
    clip_min: float
    clip_max: float
    apply_lee_filter: bool
    lee_window_size: int

@dataclass
class STURMPreprocessConfig:
    """Preprocessing parameters for STURM SAR imagery"""
    already_normalised: bool

@dataclass
class PreprocessingConfig:
    """Preprocessing settings for all datasets."""
    sen1floods11: Sen1Floods11PreprocessConfig
    sturm: STURMPreprocessConfig

@dataclass
class RandomHFlipConfig:
    enabled: bool
    p: float

@dataclass
class RandomVFlipConfig:
    enabled: bool
    p: float

@dataclass
class RandomRotation90Config:
    enabled: bool
    p: float

@dataclass
class RandomBrightnessContrastConfig:
    enabled: bool
    p: float
    brightness_limit: float
    contrast_limit: float

@dataclass
class TransformsConfig:
    random_hflip: RandomHFlipConfig
    random_vflip: RandomVFlipConfig
    random_rotation90: RandomRotation90Config
    random_brightness_contrast: RandomBrightnessContrastConfig

@dataclass
class ModelConfig:
    """Model architecture settings."""
    encoder_name: str
    encoder_weights: str
    classes: int

@dataclass
class LRSchedulerConfig:
    enabled: bool
    factor: float
    patience: int
    min_lr: float

@dataclass
class TrainConfig:
    """Training loop and optimiser settings."""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    optimiser: str
    lr_scheduler: LRSchedulerConfig
    num_workers: int
    freeze_encoder_epochs: int

@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    num_vis_samples: int

@dataclass
class InferenceConfig:
    """Inference Settings"""
    tile_size: int
    stride: int
    threshold: float

@dataclass
class SplitsConfig:
    """Splits settings."""
    test_events: list
    val_fraction: float
    random_seed: int

@dataclass
class LabelMapsConfig:
    """Label remapping for each dataset"""
    sturm: dict
    sen1floods11: dict

@dataclass
class Config:
    """Top-level config object."""
    paths: PathsConfig
    preprocessing: PreprocessingConfig
    transforms: TransformsConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    splits: SplitsConfig
    label_maps: LabelMapsConfig

# -- Loader --
def load_config(path: str | Path) -> Config:
    """
    Load a YAML config file.
    Args:
        path: Path to the YAML configuration file

    Returns:
        config: Configuration object.

    """
    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Resolve relative paths against the project root
    base = config_path.parent.parent

    d = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # paths
    p = d["paths"]
    paths = PathsConfig(
        project_root=base,
        raw_dir=(base / p["raw_dir"]).resolve(),
        processed_dir=(base / p["processed_dir"]).resolve(),
        checkpoints_dir=(base / p["checkpoints_dir"]).resolve(),
        inference_checkpoint=(base / p["inference_checkpoint"]).resolve(),
        predictions=(base / p["predictions"]).resolve(),
        figures=(base / p["figures"]).resolve(),
        logs=(base / p["logs"]).resolve(),
        splits_file=(base / p["splits_file"]).resolve(),
        inference_examples_dir=(base / p["inference_examples_dir"]).resolve(),
    )

    # preprocessing
    pre = d["preprocessing"]
    preprocessing = PreprocessingConfig(
        sen1floods11=Sen1Floods11PreprocessConfig(
            nan_fill=float(pre["sen1floods11"]["nan_fill"]),
            clip_min=float(pre["sen1floods11"]["clip_min"]),
            clip_max=float(pre["sen1floods11"]["clip_max"]),
            apply_lee_filter=bool(pre["sen1floods11"]["apply_lee_filter"]),
            lee_window_size=int(pre["sen1floods11"]["lee_window_size"]),
        ),
        sturm=STURMPreprocessConfig(
            already_normalised=bool(pre["sturm"]["already_normalised"]),
        )
    )

    # transforms
    tr = d["transforms"]
    transforms = TransformsConfig(
        random_hflip=RandomHFlipConfig(
            enabled=bool(tr["random_hflip"]["enabled"]),
            p=float(tr["random_hflip"]["p"]),
        ),
        random_vflip=RandomVFlipConfig(
            enabled=bool(tr["random_vflip"]["enabled"]),
            p=float(tr["random_vflip"]["p"]),
        ),
        random_rotation90=RandomRotation90Config(
            enabled=bool(tr["random_rotation90"]["enabled"]),
            p=float(tr["random_rotation90"]["p"]),
        ),
        random_brightness_contrast=RandomBrightnessContrastConfig(
            enabled=bool(tr["random_brightness_contrast"]["enabled"]),
            p=float(tr["random_brightness_contrast"]["p"]),
            brightness_limit=float(tr["random_brightness_contrast"]["brightness_limit"]),
            contrast_limit=float(tr["random_brightness_contrast"]["contrast_limit"]),
        ),
    )

    # model
    m = d["model"]
    model = ModelConfig(
        encoder_name=str(m["encoder_name"]),
        encoder_weights=str(m["encoder_weights"]),
        classes=int(m["classes"])
    )

    # training
    t = d["train"]
    lr_sched = t["lr_scheduler"]
    train = TrainConfig(
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        patience=int(t["patience"]),
        optimiser=str(t["optimiser"]),
        lr_scheduler=LRSchedulerConfig(
            enabled=bool(lr_sched["enabled"]),
            factor=float(lr_sched["factor"]),
            patience=int(lr_sched["patience"]),
            min_lr=float(lr_sched["min_lr"]),
        ),
        num_workers=int(t["num_workers"]),
        freeze_encoder_epochs=int(t["freeze_encoder_epochs"]),
    )

    # evaluation
    e = d["evaluation"]
    evaluation = EvaluationConfig(
        num_vis_samples=int(e["num_vis_samples"]),
    )

    # inference
    i = d["inference"]
    inference = InferenceConfig(
        tile_size=int(i["tile_size"]),
        stride=int(i["stride"]),
        threshold=float(i["threshold"]),
    )

    # splits
    s = d["splits"]
    splits = SplitsConfig(
        test_events=s["test_events"],
        val_fraction=float(s["val_fraction"]),
        random_seed=int(s["random_seed"]),
    )

    # label_maps
    lm = d["label_maps"]
    label_maps = LabelMapsConfig(
        sturm={int(k): int(v) for k, v in lm["sturm"].items()},
        sen1floods11={int(k): int(v) for k, v in lm["sen1floods11"].items()},
    )

    config = Config(
        paths=paths,
        preprocessing=preprocessing,
        transforms=transforms,
        model=model,
        evaluation=evaluation,
        inference=inference,
        train=train,
        splits=splits,
        label_maps=label_maps
    )

    return config
