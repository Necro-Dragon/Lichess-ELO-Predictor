"""Train and export a sparse-game neural rating-band baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .sparse_snapshot_corpus import (
    SPLIT_ID_TO_NAME,
    SPLIT_NAME_TO_ID,
    RatingBandSpec,
    load_sparse_snapshot_arrays,
    resolve_corpus_dir,
)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - dependency availability depends on environment
    torch = None
    F = None
    nn = None
    pack_padded_sequence = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no branch - tiny import flag helper
    TORCH_IMPORT_ERROR = None


BACKGROUND = "#faf7f2"
SURFACE = "#fffdf9"
TEXT = "#18212f"
MUTED = "#667085"
GRID = "#d8d2c8"
TRAIN_COLOR = "#d96b38"
VAL_COLOR = "#7f8ea3"
TEST_COLOR = "#2e6f95"


if torch is not None:

    class SparseSnapshotDataset(Dataset):
        """In-memory dataset backed by a sparse snapshot corpus bundle."""

        def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
            self.arrays = arrays
            self.indices = np.ascontiguousarray(indices.astype(np.int64, copy=False))

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, item: int) -> dict[str, Any]:
            game_row = int(self.indices[item])
            start = int(self.arrays["offsets"][game_row])
            length = int(self.arrays["lengths"][game_row])
            stop = start + length
            return {
                "board_codes": self.arrays["board_codes"][start:stop],
                "side_to_move": self.arrays["side_to_move"][start:stop],
                "castling_rights": self.arrays["castling_rights"][start:stop],
                "en_passant_file": self.arrays["en_passant_file"][start:stop],
                "snapshot_plies": self.arrays["snapshot_plies"][start:stop],
                "length": length,
                "final_plies": int(self.arrays["final_plies"][game_row]),
                "game_index": int(self.arrays["source_game_index"][game_row]),
                "white_elo": int(self.arrays["white_elo"][game_row]),
                "black_elo": int(self.arrays["black_elo"][game_row]),
                "white_band": int(self.arrays["white_band"][game_row]),
                "black_band": int(self.arrays["black_band"][game_row]),
                "split_id": int(self.arrays["split_id"][game_row]),
                "site": str(self.arrays["site"][game_row]),
                "opening": str(self.arrays["opening"][game_row]),
                "eco": str(self.arrays["eco"][game_row]),
            }


    class SparseGameRatingBandModel(nn.Module):
        """CNN + GRU game encoder with dual classification and regression heads."""

        def __init__(
            self,
            *,
            band_count: int,
            snapshot_embedding_dim: int = 128,
            hidden_size: int = 128,
            normalized_ply_buckets: int = 32,
        ):
            super().__init__()
            self.band_count = band_count
            self.snapshot_embedding_dim = snapshot_embedding_dim
            self.hidden_size = hidden_size
            self.normalized_ply_buckets = normalized_ply_buckets

            self.snapshot_encoder = nn.Sequential(
                nn.Conv2d(18, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.snapshot_projection = nn.Linear(64, snapshot_embedding_dim)
            self.ply_embedding = nn.Embedding(normalized_ply_buckets, snapshot_embedding_dim)
            self.sequence_encoder = nn.GRU(
                input_size=snapshot_embedding_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.white_band_head = nn.Linear(hidden_size, band_count)
            self.black_band_head = nn.Linear(hidden_size, band_count)
            self.white_elo_head = nn.Linear(hidden_size, 1)
            self.black_elo_head = nn.Linear(hidden_size, 1)

        def _board_codes_to_planes(
            self,
            board_codes: torch.Tensor,
            side_to_move: torch.Tensor,
            castling_rights: torch.Tensor,
            en_passant_file: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, sequence_length, _flat = board_codes.shape
            board_grid = board_codes.view(batch_size, sequence_length, 8, 8)
            piece_planes = [(board_grid == piece_code).float() for piece_code in range(1, 13)]
            side_plane = side_to_move.float().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
            castling_planes = [
                castling_rights[:, :, plane_index].float().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
                for plane_index in range(4)
            ]

            ep_plane = torch.zeros(
                (batch_size, sequence_length, 8, 8),
                dtype=torch.float32,
                device=board_codes.device,
            )
            active_ep = en_passant_file > 0
            if torch.any(active_ep):
                active_batch, active_seq = active_ep.nonzero(as_tuple=True)
                file_index = en_passant_file[active_ep].long() - 1
                ep_plane[active_batch, active_seq, :, file_index] = 1.0

            all_planes = torch.stack(piece_planes + [side_plane, *castling_planes, ep_plane], dim=2)
            return all_planes

        def encode_games(self, batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
            board_codes = batch["board_codes"]
            side_to_move = batch["side_to_move"]
            castling_rights = batch["castling_rights"]
            en_passant_file = batch["en_passant_file"]
            snapshot_plies = batch["snapshot_plies"]
            final_plies = batch["final_plies"]
            lengths = batch["lengths"]

            planes = self._board_codes_to_planes(
                board_codes=board_codes,
                side_to_move=side_to_move,
                castling_rights=castling_rights,
                en_passant_file=en_passant_file,
            )
            batch_size, sequence_length, plane_count, height, width = planes.shape
            encoded = self.snapshot_encoder(planes.view(batch_size * sequence_length, plane_count, height, width))
            encoded = encoded.view(batch_size * sequence_length, -1)
            snapshot_vectors = self.snapshot_projection(encoded).view(batch_size, sequence_length, -1)

            final_plies_safe = final_plies.clamp(min=1).float().unsqueeze(1)
            normalized = snapshot_plies.float() / final_plies_safe
            normalized = torch.clamp(normalized, min=0.0, max=1.0)
            bucket_ids = torch.round(normalized * (self.normalized_ply_buckets - 1)).long()
            snapshot_vectors = snapshot_vectors + self.ply_embedding(bucket_ids)

            packed = pack_padded_sequence(
                snapshot_vectors,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _output, hidden = self.sequence_encoder(packed)
            return hidden[-1]

        def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
            embedding = self.encode_games(batch)
            return {
                "embedding": embedding,
                "white_band_logits": self.white_band_head(embedding),
                "black_band_logits": self.black_band_head(embedding),
                "white_elo_pred": self.white_elo_head(embedding).squeeze(-1),
                "black_elo_pred": self.black_elo_head(embedding).squeeze(-1),
            }


else:  # pragma: no cover - used only when torch is unavailable

    class SparseSnapshotDataset:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any):
            raise ImportError(f"PyTorch is required for training support: {TORCH_IMPORT_ERROR}")


    class SparseGameRatingBandModel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any):
            raise ImportError(f"PyTorch is required for training support: {TORCH_IMPORT_ERROR}")


@dataclass(slots=True)
class TrainingConfig:
    corpus_path: str
    checkpoint_dir: str
    graphics_dir: str
    report_path: str
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size_accelerator: int = 128
    batch_size_cpu: int = 64
    gradient_clip_norm: float = 1.0
    regression_loss_weight: float = 0.25
    normalized_ply_buckets: int = 32
    snapshot_embedding_dim: int = 128
    hidden_size: int = 128
    seed: int = 20260416
    class_weight_clip_min: float = 0.25
    class_weight_clip_max: float = 4.0


def _require_torch() -> None:
    if torch is None:
        raise ImportError(f"PyTorch is required for this command: {TORCH_IMPORT_ERROR}")


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/rapid_2026-03_sparse_snapshot_200000", help="Sparse-snapshot corpus directory.")
    parser.add_argument("--checkpoint-dir", default="artifacts/rating_band_training", help="Directory for checkpoints and JSON artifacts.")
    parser.add_argument("--graphics-dir", default="report/assets", help="Directory for training SVG charts.")
    parser.add_argument("--report-path", default="report/rapid_sparse_rating_band_training_report.md", help="Path to the generated markdown report.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=4, help="Early-stopping patience on validation loss.")
    parser.add_argument("--seed", type=int, default=20260416, help="Training RNG seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--batch-size-accelerator", type=int, default=128, help="Batch size on MPS or CUDA.")
    parser.add_argument("--batch-size-cpu", type=int, default=64, help="Batch size on CPU.")
    return parser.parse_args(argv)


def parse_export_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load.")
    parser.add_argument("--corpus", default="data/rapid_2026-03_sparse_snapshot_200000", help="Sparse-snapshot corpus directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to encode.")
    parser.add_argument("--out", required=True, help="Output .npz file for embeddings.")
    return parser.parse_args(argv)


def select_device() -> "torch.device":
    _require_torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_accelerator_device(device: "torch.device") -> bool:
    return device.type in {"mps", "cuda"}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_split_indices(arrays: dict[str, np.ndarray], split_name: str) -> np.ndarray:
    if split_name == "all":
        return np.arange(len(arrays["split_id"]), dtype=np.int64)
    split_id = SPLIT_NAME_TO_ID[split_name]
    return np.flatnonzero(arrays["split_id"] == split_id).astype(np.int64)


def collate_sparse_snapshot_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_length = max(int(sample["length"]) for sample in samples)

    board_codes = np.zeros((batch_size, max_length, 64), dtype=np.uint8)
    side_to_move = np.zeros((batch_size, max_length), dtype=np.uint8)
    castling_rights = np.zeros((batch_size, max_length, 4), dtype=np.uint8)
    en_passant_file = np.zeros((batch_size, max_length), dtype=np.uint8)
    snapshot_plies = np.zeros((batch_size, max_length), dtype=np.uint16)
    lengths = np.zeros(batch_size, dtype=np.int64)
    final_plies = np.zeros(batch_size, dtype=np.int64)
    game_index = np.zeros(batch_size, dtype=np.int64)
    white_elo = np.zeros(batch_size, dtype=np.float32)
    black_elo = np.zeros(batch_size, dtype=np.float32)
    white_band = np.zeros(batch_size, dtype=np.int64)
    black_band = np.zeros(batch_size, dtype=np.int64)
    split_id = np.zeros(batch_size, dtype=np.int64)
    site: list[str] = []
    opening: list[str] = []
    eco: list[str] = []

    for row, sample in enumerate(samples):
        length = int(sample["length"])
        board_codes[row, :length] = sample["board_codes"]
        side_to_move[row, :length] = sample["side_to_move"]
        castling_rights[row, :length] = sample["castling_rights"]
        en_passant_file[row, :length] = sample["en_passant_file"]
        snapshot_plies[row, :length] = sample["snapshot_plies"]
        lengths[row] = length
        final_plies[row] = int(sample["final_plies"])
        game_index[row] = int(sample["game_index"])
        white_elo[row] = float(sample["white_elo"])
        black_elo[row] = float(sample["black_elo"])
        white_band[row] = int(sample["white_band"])
        black_band[row] = int(sample["black_band"])
        split_id[row] = int(sample["split_id"])
        site.append(sample["site"])
        opening.append(sample["opening"])
        eco.append(sample["eco"])

    return {
        "board_codes": torch.from_numpy(board_codes),
        "side_to_move": torch.from_numpy(side_to_move),
        "castling_rights": torch.from_numpy(castling_rights),
        "en_passant_file": torch.from_numpy(en_passant_file),
        "snapshot_plies": torch.from_numpy(snapshot_plies),
        "lengths": torch.from_numpy(lengths),
        "final_plies": torch.from_numpy(final_plies),
        "game_index": torch.from_numpy(game_index),
        "white_elo": torch.from_numpy(white_elo),
        "black_elo": torch.from_numpy(black_elo),
        "white_band": torch.from_numpy(white_band),
        "black_band": torch.from_numpy(black_band),
        "split_id": torch.from_numpy(split_id),
        "site": site,
        "opening": opening,
        "eco": eco,
    }


def move_batch_to_device(batch: dict[str, Any], device: "torch.device") -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_class_weights(
    arrays: dict[str, np.ndarray],
    *,
    class_count: int,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    train_indices = get_split_indices(arrays, "train")
    white_counts = np.bincount(arrays["white_band"][train_indices], minlength=class_count)
    black_counts = np.bincount(arrays["black_band"][train_indices], minlength=class_count)
    counts = white_counts + black_counts
    safe_counts = np.maximum(counts, 1)
    raw = counts.sum() / (class_count * safe_counts.astype(np.float32))
    return np.clip(raw, clip_min, clip_max).astype(np.float32)


def build_elo_normalization(arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    train_indices = get_split_indices(arrays, "train")
    train_ratings = np.concatenate(
        [arrays["white_elo"][train_indices], arrays["black_elo"][train_indices]]
    ).astype(np.float32)
    mean = float(train_ratings.mean())
    std = float(train_ratings.std())
    if math.isclose(std, 0.0):
        std = 1.0
    return mean, std


def build_dataloader(
    arrays: dict[str, np.ndarray],
    *,
    split_name: str,
    batch_size: int,
    shuffle: bool,
) -> "DataLoader":
    indices = get_split_indices(arrays, split_name)
    dataset = SparseSnapshotDataset(arrays, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_sparse_snapshot_batch,
    )


def compute_loss_and_metrics(
    outputs: dict[str, "torch.Tensor"],
    batch: dict[str, Any],
    *,
    class_weights: "torch.Tensor",
    elo_mean: float,
    elo_std: float,
    regression_weight: float,
) -> tuple["torch.Tensor", dict[str, float]]:
    white_ce = F.cross_entropy(outputs["white_band_logits"], batch["white_band"], weight=class_weights)
    black_ce = F.cross_entropy(outputs["black_band_logits"], batch["black_band"], weight=class_weights)

    white_target_z = (batch["white_elo"] - elo_mean) / elo_std
    black_target_z = (batch["black_elo"] - elo_mean) / elo_std
    white_reg = F.smooth_l1_loss(outputs["white_elo_pred"], white_target_z)
    black_reg = F.smooth_l1_loss(outputs["black_elo_pred"], black_target_z)

    classification_loss = white_ce + black_ce
    regression_loss = white_reg + black_reg
    total_loss = classification_loss + (regression_weight * regression_loss)

    white_pred = outputs["white_band_logits"].argmax(dim=1)
    black_pred = outputs["black_band_logits"].argmax(dim=1)
    white_correct = int((white_pred == batch["white_band"]).sum().item())
    black_correct = int((black_pred == batch["black_band"]).sum().item())
    player_count = int(batch["white_band"].numel() * 2)

    white_elo = outputs["white_elo_pred"] * elo_std + elo_mean
    black_elo = outputs["black_elo_pred"] * elo_std + elo_mean
    abs_error = torch.abs(white_elo - batch["white_elo"]).sum() + torch.abs(black_elo - batch["black_elo"]).sum()
    sq_error = torch.square(white_elo - batch["white_elo"]).sum() + torch.square(black_elo - batch["black_elo"]).sum()

    return total_loss, {
        "total_loss": float(total_loss.item()),
        "classification_loss": float(classification_loss.item()),
        "regression_loss": float(regression_loss.item()),
        "white_correct": float(white_correct),
        "black_correct": float(black_correct),
        "player_count": float(player_count),
        "abs_error_sum": float(abs_error.item()),
        "sq_error_sum": float(sq_error.item()),
        "batch_size": float(batch["white_band"].shape[0]),
    }


def aggregate_metric_batches(metric_batches: list[dict[str, float]]) -> dict[str, float]:
    if not metric_batches:
        return {
            "total_loss": math.nan,
            "classification_loss": math.nan,
            "regression_loss": math.nan,
            "band_accuracy": math.nan,
            "white_accuracy": math.nan,
            "black_accuracy": math.nan,
            "elo_mae": math.nan,
            "elo_rmse": math.nan,
        }

    batch_count = sum(batch["batch_size"] for batch in metric_batches)
    player_count = sum(batch["player_count"] for batch in metric_batches)
    white_player_count = player_count / 2.0
    return {
        "total_loss": sum(batch["total_loss"] * batch["batch_size"] for batch in metric_batches) / batch_count,
        "classification_loss": sum(batch["classification_loss"] * batch["batch_size"] for batch in metric_batches) / batch_count,
        "regression_loss": sum(batch["regression_loss"] * batch["batch_size"] for batch in metric_batches) / batch_count,
        "band_accuracy": (sum(batch["white_correct"] + batch["black_correct"] for batch in metric_batches) / player_count),
        "white_accuracy": sum(batch["white_correct"] for batch in metric_batches) / white_player_count,
        "black_accuracy": sum(batch["black_correct"] for batch in metric_batches) / white_player_count,
        "elo_mae": sum(batch["abs_error_sum"] for batch in metric_batches) / player_count,
        "elo_rmse": math.sqrt(sum(batch["sq_error_sum"] for batch in metric_batches) / player_count),
    }


def _iterate_loader(
    loader: "DataLoader",
    *,
    description: str,
    show_progress: bool,
) -> Any:
    if not show_progress:
        return loader
    return tqdm(loader, desc=description, unit="batch", leave=False)


def run_epoch(
    *,
    model: "SparseGameRatingBandModel",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer | None",
    device: "torch.device",
    class_weights: "torch.Tensor",
    elo_mean: float,
    elo_std: float,
    regression_weight: float,
    gradient_clip_norm: float,
    description: str,
    show_progress: bool,
) -> dict[str, float]:
    metric_batches: list[dict[str, float]] = []
    is_training = optimizer is not None
    model.train(is_training)

    iterator = _iterate_loader(loader, description=description, show_progress=show_progress)
    for raw_batch in iterator:
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(is_training):
            outputs = model(batch)
            loss, batch_metrics = compute_loss_and_metrics(
                outputs,
                batch,
                class_weights=class_weights,
                elo_mean=elo_mean,
                elo_std=elo_std,
                regression_weight=regression_weight,
            )
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        metric_batches.append(batch_metrics)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{batch_metrics['total_loss']:.3f}")

    if show_progress and hasattr(iterator, "close"):
        iterator.close()
    return aggregate_metric_batches(metric_batches)


def make_svg_root(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{title}</desc>
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#b8ab95" flood-opacity="0.18"/>
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="{BACKGROUND}" />
{body}
</svg>
"""


def _project_value(value: float, lower: float, upper: float, start: float, span: float) -> float:
    if math.isclose(lower, upper):
        return start + (span / 2.0)
    return start + ((value - lower) / (upper - lower)) * span


def render_metric_chart_svg(
    history: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    y_label: str,
) -> str:
    width, height = 1600, 980
    chart_x = 120
    chart_y = 130
    chart_w = 1320
    chart_h = 720

    epochs = [entry["epoch"] for entry in history]
    train_values = [float(entry["train"][metric_key]) for entry in history]
    val_values = [float(entry["val"][metric_key]) for entry in history]
    test_values = [float(entry["test"][metric_key]) for entry in history]

    all_values = [value for value in train_values + val_values + test_values if not math.isnan(value)]
    min_y = min(all_values)
    max_y = max(all_values)
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0

    grid_lines: list[str] = []
    y_ticks: list[str] = []
    for step in range(6):
        ratio = step / 5
        y = chart_y + chart_h - (chart_h * ratio)
        tick_value = min_y + ((max_y - min_y) * ratio)
        grid_lines.append(
            f'  <line x1="{chart_x}" y1="{y:.2f}" x2="{chart_x + chart_w}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )
        y_ticks.append(
            f'  <text x="{chart_x - 20}" y="{y + 8:.2f}" text-anchor="end" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{tick_value:.3f}</text>'
        )

    x_ticks: list[str] = []
    for epoch in epochs:
        x = _project_value(float(epoch), float(min(epochs)), float(max(epochs)), chart_x, chart_w)
        x_ticks.append(
            f'  <line x1="{x:.2f}" y1="{chart_y + chart_h}" x2="{x:.2f}" y2="{chart_y + chart_h + 12}" stroke="{TEXT}" stroke-width="1"/>'
            f'  <text x="{x:.2f}" y="{chart_y + chart_h + 40}" text-anchor="middle" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{epoch}</text>'
        )

    def line_path(values: list[float]) -> str:
        segments = []
        for epoch, value in zip(epochs, values, strict=True):
            x = _project_value(float(epoch), float(min(epochs)), float(max(epochs)), chart_x, chart_w)
            y = chart_y + chart_h - _project_value(value, min_y, max_y, 0, chart_h)
            command = "M" if not segments else "L"
            segments.append(f"{command} {x:.2f} {y:.2f}")
        return " ".join(segments)

    legend = [
        (TRAIN_COLOR, "Train", ""),
        (VAL_COLOR, "Validation", ' stroke-dasharray="10 8"'),
        (TEST_COLOR, "Test", ""),
    ]
    legend_items = []
    for index, (color, label, dash) in enumerate(legend):
        y = 88 + index * 34
        legend_items.append(
            f'  <line x1="1220" y1="{y}" x2="1284" y2="{y}" stroke="{color}" stroke-width="5"{dash}/>'
            f'  <text x="1300" y="{y + 8}" fill="{TEXT}" font-size="22" font-family="Georgia, serif">{label}</text>'
        )

    body = "\n".join(
        [
            '  <g filter="url(#shadow)">',
            f'    <rect x="70" y="42" width="{width - 140}" height="{height - 84}" rx="28" fill="{SURFACE}"/>',
            "  </g>",
            f'  <text x="{chart_x}" y="94" fill="{TEXT}" font-size="42" font-weight="700" font-family="Georgia, serif">{title}</text>',
            *legend_items,
            *grid_lines,
            f'  <line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="2"/>',
            f'  <line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="2"/>',
            f'  <path d="{line_path(train_values)}" fill="none" stroke="{TRAIN_COLOR}" stroke-width="5"/>',
            f'  <path d="{line_path(val_values)}" fill="none" stroke="{VAL_COLOR}" stroke-width="4" stroke-dasharray="10 8"/>',
            f'  <path d="{line_path(test_values)}" fill="none" stroke="{TEST_COLOR}" stroke-width="5"/>',
            *x_ticks,
            *y_ticks,
            f'  <text x="{chart_x + chart_w / 2:.2f}" y="{height - 48}" text-anchor="middle" fill="{MUTED}" font-size="24" font-family="Georgia, serif">Epoch</text>',
            f'  <text x="48" y="{chart_y + chart_h / 2:.2f}" transform="rotate(-90 48 {chart_y + chart_h / 2:.2f})" text-anchor="middle" fill="{MUTED}" font-size="24" font-family="Georgia, serif">{y_label}</text>',
        ]
    )
    return make_svg_root(width, height, title, body)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_report_markdown(
    *,
    history: list[dict[str, Any]],
    summary: dict[str, Any],
    training_config: TrainingConfig,
    corpus_manifest: dict[str, Any],
    loss_chart_path: Path,
    accuracy_chart_path: Path,
    report_path: Path,
) -> str:
    best_epoch = int(summary["best_epoch"])
    final_test = summary["best_test_metrics"]
    loss_rel = os.path.relpath(loss_chart_path, report_path.parent)
    accuracy_rel = os.path.relpath(accuracy_chart_path, report_path.parent)

    return "\n".join(
        [
            "---",
            'title: "Sparse-Game Neural Rating-Band Training Report"',
            f'date: "{datetime.now().date().isoformat()}"',
            "fontsize: 10pt",
            "---",
            "",
            "# Summary",
            "",
            f"This run trained on a uniform `{corpus_manifest['selected_game_count']:,}`-game subset of the March 2026 rapid sample using sparse board snapshots every `{corpus_manifest['snapshot_step']}` plies with the final position appended once when needed.",
            f"The model predicts both White and Black `200`-point rating bands while also learning auxiliary exact-Elo regression targets. The shared GRU state is the exported `{training_config.hidden_size}`-dimensional game embedding.",
            "",
            "## Data And Model",
            "",
            f"- Source subset size: `{corpus_manifest['selected_game_count']:,}` games",
            f"- Train/val/test split: `{corpus_manifest['split_counts']['train']:,}` / `{corpus_manifest['split_counts']['val']:,}` / `{corpus_manifest['split_counts']['test']:,}` games",
            f"- Total snapshots: `{corpus_manifest['total_snapshot_count']:,}`",
            f"- Max snapshots per game: `{corpus_manifest['max_snapshots_per_game']}`",
            f"- Rating bands: `{', '.join(corpus_manifest['rating_bands']['labels'])}`",
            f"- Optimizer: `AdamW(lr={training_config.learning_rate}, weight_decay={training_config.weight_decay})`",
            f"- Early stopping patience: `{training_config.patience}` epochs",
            "",
            "## Training Curves",
            "",
            f"![Training and test total loss]({loss_rel})",
            "",
            f"![Training and test band accuracy]({accuracy_rel})",
            "",
            "## Final Held-Out Metrics",
            "",
            f"- Best validation epoch: `{best_epoch}`",
            f"- Test total loss at best epoch: `{final_test['total_loss']:.4f}`",
            f"- Test band accuracy at best epoch: `{final_test['band_accuracy'] * 100:.2f}%`",
            f"- Test exact-Elo MAE at best epoch: `{final_test['elo_mae']:.2f}`",
            f"- Test exact-Elo RMSE at best epoch: `{final_test['elo_rmse']:.2f}`",
            "",
            "## Embedding Usage",
            "",
            "The trained checkpoint exposes `encode_games(batch)` through `SparseGameRatingBandModel`, returning one `128`-dimensional embedding per game. The `export_game_embeddings` command writes those embeddings to an `.npz` file alongside `game_index`, Elo labels, band ids, split labels, and copied metadata such as `site` and `opening`.",
        ]
    ) + "\n"


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model: "SparseGameRatingBandModel",
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    model_config: dict[str, Any],
    training_config: TrainingConfig,
    band_spec: RatingBandSpec,
    class_weights: np.ndarray,
    elo_mean: float,
    elo_std: float,
    corpus_manifest: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_config": model_config,
        "training_config": asdict(training_config),
        "band_spec": band_spec.to_dict(),
        "class_weights": class_weights.tolist(),
        "elo_mean": elo_mean,
        "elo_std": elo_std,
        "corpus_manifest_path": corpus_manifest.get("source_manifest_path", ""),
    }
    torch.save(payload, checkpoint_path)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: "torch.device",
) -> tuple["SparseGameRatingBandModel", dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = SparseGameRatingBandModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def train_rating_band_model(
    *,
    corpus_path: Path,
    checkpoint_dir: Path,
    graphics_dir: Path,
    report_path: Path,
    epochs: int,
    patience: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size_accelerator: int,
    batch_size_cpu: int,
) -> dict[str, Path]:
    _require_torch()
    set_random_seed(seed)
    device = select_device()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    graphics_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_manifest, arrays = load_sparse_snapshot_arrays(corpus_path)
    band_spec = RatingBandSpec.from_dict(corpus_manifest["rating_bands"])
    batch_size = batch_size_accelerator if is_accelerator_device(device) else batch_size_cpu

    training_config = TrainingConfig(
        corpus_path=str(resolve_corpus_dir(corpus_path).resolve()),
        checkpoint_dir=str(checkpoint_dir.resolve()),
        graphics_dir=str(graphics_dir.resolve()),
        report_path=str(report_path.resolve()),
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size_accelerator=batch_size_accelerator,
        batch_size_cpu=batch_size_cpu,
        seed=seed,
    )

    train_loader = build_dataloader(arrays, split_name="train", batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(arrays, split_name="val", batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(arrays, split_name="test", batch_size=batch_size, shuffle=False)

    model_config = {
        "band_count": band_spec.class_count,
        "snapshot_embedding_dim": training_config.snapshot_embedding_dim,
        "hidden_size": training_config.hidden_size,
        "normalized_ply_buckets": training_config.normalized_ply_buckets,
    }
    model = SparseGameRatingBandModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    class_weights_np = build_class_weights(
        arrays,
        class_count=band_spec.class_count,
        clip_min=training_config.class_weight_clip_min,
        clip_max=training_config.class_weight_clip_max,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    elo_mean, elo_std = build_elo_normalization(arrays)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_loss = math.inf
    stale_epochs = 0
    epoch_paths: list[Path] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            elo_mean=elo_mean,
            elo_std=elo_std,
            regression_weight=training_config.regression_loss_weight,
            gradient_clip_norm=training_config.gradient_clip_norm,
            description=f"Epoch {epoch}/{epochs} train",
            show_progress=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            class_weights=class_weights,
            elo_mean=elo_mean,
            elo_std=elo_std,
            regression_weight=training_config.regression_loss_weight,
            gradient_clip_norm=training_config.gradient_clip_norm,
            description=f"Epoch {epoch}/{epochs} val",
            show_progress=True,
        )

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            model_config=model_config,
            training_config=training_config,
            band_spec=band_spec,
            class_weights=class_weights_np,
            elo_mean=elo_mean,
            elo_std=elo_std,
            corpus_manifest=corpus_manifest,
        )
        epoch_paths.append(checkpoint_path)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "test": {
                    "total_loss": math.nan,
                    "classification_loss": math.nan,
                    "regression_loss": math.nan,
                    "band_accuracy": math.nan,
                    "white_accuracy": math.nan,
                    "black_accuracy": math.nan,
                    "elo_mae": math.nan,
                    "elo_rmse": math.nan,
                },
            }
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            stale_epochs = 0
            shutil.copy2(checkpoint_path, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    replay_progress = tqdm(epoch_paths, desc="Replaying checkpoints on test split", unit="checkpoint")
    for epoch_path in replay_progress:
        replay_model, checkpoint = load_model_from_checkpoint(epoch_path, device=device)
        test_metrics = run_epoch(
            model=replay_model,
            loader=test_loader,
            optimizer=None,
            device=device,
            class_weights=class_weights,
            elo_mean=elo_mean,
            elo_std=elo_std,
            regression_weight=training_config.regression_loss_weight,
            gradient_clip_norm=training_config.gradient_clip_norm,
            description=f"Test epoch {checkpoint['epoch']}",
            show_progress=False,
        )
        history[int(checkpoint["epoch"]) - 1]["test"] = test_metrics
        replay_progress.set_postfix(loss=f"{test_metrics['total_loss']:.3f}")
    replay_progress.close()

    history_path = checkpoint_dir / "history.json"
    write_json(
        history_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "epochs": history,
        },
    )

    best_test_metrics = history[best_epoch - 1]["test"]
    metrics_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_train_metrics": history[best_epoch - 1]["train"],
        "best_validation_metrics": history[best_epoch - 1]["val"],
        "best_test_metrics": best_test_metrics,
        "completed_epochs": len(history),
    }
    metrics_summary_path = checkpoint_dir / "metrics_summary.json"
    write_json(metrics_summary_path, metrics_summary)

    run_metadata_path = checkpoint_dir / "run_metadata.json"
    write_json(
        run_metadata_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": str(device),
            "training_config": asdict(training_config),
            "rating_bands": band_spec.to_dict(),
            "class_weights": class_weights_np.tolist(),
            "elo_mean": elo_mean,
            "elo_std": elo_std,
            "corpus_path": str(resolve_corpus_dir(corpus_path).resolve()),
        },
    )

    loss_chart_path = graphics_dir / "rating_band_training_loss.svg"
    accuracy_chart_path = graphics_dir / "rating_band_training_accuracy.svg"
    loss_chart_path.write_text(
        render_metric_chart_svg(
            history,
            metric_key="total_loss",
            title="Sparse-game training loss over epochs",
            y_label="Total loss",
        ),
        encoding="utf-8",
    )
    accuracy_chart_path.write_text(
        render_metric_chart_svg(
            history,
            metric_key="band_accuracy",
            title="Sparse-game band accuracy over epochs",
            y_label="Band accuracy",
        ),
        encoding="utf-8",
    )

    report_text = build_report_markdown(
        history=history,
        summary=metrics_summary,
        training_config=training_config,
        corpus_manifest=corpus_manifest,
        loss_chart_path=loss_chart_path,
        accuracy_chart_path=accuracy_chart_path,
        report_path=report_path,
    )
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "history_json": history_path,
        "metrics_summary_json": metrics_summary_path,
        "run_metadata_json": run_metadata_path,
        "best_checkpoint": checkpoint_dir / "best.pt",
        "loss_chart_svg": loss_chart_path,
        "accuracy_chart_svg": accuracy_chart_path,
        "report_md": report_path,
    }


def export_game_embeddings(
    *,
    checkpoint_path: Path,
    corpus_path: Path,
    split_name: str,
    out_path: Path,
) -> Path:
    _require_torch()
    device = select_device()
    _manifest, arrays = load_sparse_snapshot_arrays(corpus_path)
    batch_size = 128 if is_accelerator_device(device) else 64
    loader = build_dataloader(arrays, split_name=split_name, batch_size=batch_size, shuffle=False)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    embeddings: list[np.ndarray] = []
    game_index: list[np.ndarray] = []
    white_elo: list[np.ndarray] = []
    black_elo: list[np.ndarray] = []
    white_band: list[np.ndarray] = []
    black_band: list[np.ndarray] = []
    split_labels: list[np.ndarray] = []
    sites: list[str] = []
    openings: list[str] = []
    ecos: list[str] = []

    progress = tqdm(loader, desc="Exporting embeddings", unit="batch")
    with torch.no_grad():
        for raw_batch in progress:
            batch = move_batch_to_device(raw_batch, device)
            encoded = model.encode_games(batch)
            embeddings.append(encoded.detach().cpu().numpy())
            game_index.append(raw_batch["game_index"].numpy())
            white_elo.append(raw_batch["white_elo"].numpy())
            black_elo.append(raw_batch["black_elo"].numpy())
            white_band.append(raw_batch["white_band"].numpy())
            black_band.append(raw_batch["black_band"].numpy())
            split_labels.append(
                np.asarray(
                    [SPLIT_ID_TO_NAME[int(split_id)] for split_id in raw_batch["split_id"].numpy()],
                    dtype="<U5",
                )
            )
            sites.extend(raw_batch["site"])
            openings.extend(raw_batch["opening"])
            ecos.extend(raw_batch["eco"])
    progress.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=np.ascontiguousarray(np.concatenate(embeddings, axis=0), dtype=np.float32),
        game_index=np.ascontiguousarray(np.concatenate(game_index, axis=0), dtype=np.int64),
        white_elo=np.ascontiguousarray(np.concatenate(white_elo, axis=0), dtype=np.float32),
        black_elo=np.ascontiguousarray(np.concatenate(black_elo, axis=0), dtype=np.float32),
        white_band=np.ascontiguousarray(np.concatenate(white_band, axis=0), dtype=np.int64),
        black_band=np.ascontiguousarray(np.concatenate(black_band, axis=0), dtype=np.int64),
        split=np.asarray(np.concatenate(split_labels, axis=0)),
        site=np.asarray(sites),
        opening=np.asarray(openings),
        eco=np.asarray(ecos),
        checkpoint_epoch=np.asarray([int(checkpoint["epoch"])]),
    )
    return out_path


def train_main(argv: list[str] | None = None) -> int:
    args = parse_train_args(argv)
    train_rating_band_model(
        corpus_path=Path(args.corpus),
        checkpoint_dir=Path(args.checkpoint_dir),
        graphics_dir=Path(args.graphics_dir),
        report_path=Path(args.report_path),
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size_accelerator=args.batch_size_accelerator,
        batch_size_cpu=args.batch_size_cpu,
    )
    return 0


def export_main(argv: list[str] | None = None) -> int:
    args = parse_export_args(argv)
    export_game_embeddings(
        checkpoint_path=Path(args.checkpoint),
        corpus_path=Path(args.corpus),
        split_name=args.split,
        out_path=Path(args.out),
    )
    return 0
