"""
evaluation_pipeline.py

Evaluation pipeline.

Loads a trained model checkpoint, runs evaluation on the test set,
computes and saves metrics and visualisations.
"""
from pathlib import Path
from typing import Optional
import torch
from sentinel_flood_mapper import Config
from sentinel_flood_mapper.data import get_dataloader, build_file_pairs
from sentinel_flood_mapper.models import get_model, evaluate
from sentinel_flood_mapper.utils import (
    load_checkpoint,
    plot_predictions,
    plot_contingency_map,
    plot_metrics_summary
)

class EvaluationPipeline:
    """
    Evaluation pipeline for assessing trained model performance on the test set.

    Loads a checkpoint, runs inference over the test dataloader, computes
    pixel-wise segmentation metrics and saves visualisation.
    """
    def __init__(
            self,
            config: Config,
            checkpoint_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            config: Configuration object
            checkpoint_path: Path to the model checkpoint to evaluate. If
                None, uses config.paths.inference_checkpoint
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path or config.paths.inference_checkpoint)

        print(f"Using device: {self.device}")
        print(f"Checkpoint: {self.checkpoint_path.relative_to(self.config.paths.project_root)}")

        self.model = None
        self.test_loader = None
        self.results = None
        self.vis_samples = None

    def load_model(self) -> None:
        """
        Build model architecture and load weights from checkpoint.
        """
        print("\n" + "=" * 70)
        print("LOADING MODEL")
        print("=" * 70)

        self.model, checkpoint = load_checkpoint(
            model=get_model(
                model_config=self.config.model,
                device=self.device,
            ),
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def create_dataloader(self) -> None:
        """
        Build the test dataloader from the splits CSV
        """
        print("\n" + "=" * 70)
        print("CREATING TEST DATALOADER")
        print("=" * 70)

        paths = self.config.paths

        test_pairs = build_file_pairs(
            splits_csv=paths.splits_file,
            split="test",
            project_root=paths.project_root
        )

        self.test_loader = get_dataloader(
            file_pairs=test_pairs,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
        )

        print(f"Test loader: {len(self.test_loader.dataset):,} tiles, "
              f"{len(self.test_loader)} batches")

    def run_evaluation(self) -> None:
        """
        Run evaluation on the test set and compute metrics.
        """
        print("\n" + "=" * 70)
        print("RUNNING EVALUATION")
        print("=" * 70)

        self.results, self.vis_samples = evaluate(
            model=self.model,
            dataloader=self.test_loader,
            device=self.device,
            num_vis_samples=self.config.evaluation.num_vis_samples,
        )

        print("\nTest set metrics:")
        print(f"IoU: {self.results['iou']:.4f}")
        print(f"F1: {self.results['f1']:.4f}")
        print(f"Precision: {self.results['precision']:.4f}")
        print(f"Recall: {self.results['recall']:.4f}")
        print(f"Accuracy: {self.results['accuracy']:.4f}")

    def save_results(self) -> None:
        """
        Save metrics summary and visualisations to outputs/figures/
        """
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        figures_dir = self.config.paths.figures
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Metrics bar chart
        save_path_metrics = figures_dir / "test_metrics.png"
        plot_metrics_summary(
            metrics=self.results,
            title="Test set evaluation metrics",
            save_path=save_path_metrics,
            show=False,
        )
        print(f"Metrics summary saved to {save_path_metrics.relative_to(self.config.paths.project_root)}")

        images_all = self.vis_samples["images"]
        labels_all = self.vis_samples["labels"]
        predictions_all = self.vis_samples["predictions"]

        #  Filter out tiles where image is mostly black
        valid_indices = []
        for i in range(images_all.shape[0]):
            mean_val = images_all[i].mean().item()
            has_flood = (labels_all[i] == 1).any().item()
            if mean_val > 0.01 and has_flood:
                valid_indices.append(i)

        # Use only valid tiles for visualisation
        if valid_indices:
            valid_images = images_all[valid_indices]
            valid_labels = labels_all[valid_indices]
            valid_predictions = predictions_all[valid_indices]

            save_path_preds = figures_dir / "test_predictions.png"
            if self.vis_samples is not None:
                plot_predictions(
                    images=valid_images,
                    labels=valid_labels,
                    predictions=valid_predictions,
                    n_samples=min(self.config.evaluation.num_vis_samples, len(valid_indices)),
                    save_path=save_path_preds,
                    show=False,
                )
                print(f"Sample predictions saved to {save_path_preds.relative_to(self.config.paths.project_root)}")

                # Prefer tiles with at least 10% flood coverage for contingency map
                best_idx = valid_indices[0]
                best_flood_pct = 0.0

                for i in valid_indices:
                    flood_pct = (labels_all[i] == 1).float().mean().item()
                    if flood_pct > best_flood_pct:
                        best_flood_pct = flood_pct
                        best_idx = i

                save_path_cont = figures_dir / "test_contingency_map.png"
                plot_contingency_map(
                    image=images_all[best_idx].numpy(),
                    label=labels_all[best_idx].numpy(),
                    prediction=torch.sigmoid(
                        predictions_all[best_idx]
                    ).squeeze(0).numpy(),
                    save_path=save_path_cont,
                    show=False,
                )
                print(f"Contingency map saved to {save_path_cont.relative_to(self.config.paths.project_root)}")

    def get_summary(self) -> None:
        """
        Print a summary of the evaluation results.
        """
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Checkpoint: {self.checkpoint_path.relative_to(self.config.paths.project_root)}")

        if self.test_loader:
            print(f"\nTest set: {len(self.test_loader.dataset):} tiles")

        if self.results:
            print(f"\nMetrics:")
            print(f"IoU: {self.results['iou']:.4f}")
            print(f"F1: {self.results['f1']:.4f}")
            print(f"Precision: {self.results['precision']:.4f}")
            print(f"Recall: {self.results['recall']:.4f}")
            print(f"Accuracy: {self.results['accuracy']:.4f}")

        print("=" * 70)

def run_evaluation_pipeline(
        config: Config,
        checkpoint_path: Optional[Path] = None,
) -> EvaluationPipeline:
    """
    Run the complete evaluation pipeline in one function call.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint. If None uses config.paths.inference_checkpoint

    Returns:
        EvaluationPipeline instance after completion
    """
    pipeline = EvaluationPipeline(
        config=config,
        checkpoint_path=checkpoint_path,
    )
    pipeline.load_model()
    pipeline.create_dataloader()
    pipeline.run_evaluation()
    pipeline.save_results()
    pipeline.get_summary()
    return pipeline










