"""Configuration defaults for the secondary graph-baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias


MixRatios: TypeAlias = tuple[tuple[str, float], ...]


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    """Small immutable config object for reproducible experiment runs."""

    model_name: str = "meta-llama/Llama-3.1-8B"
    device: str | None = None
    candidate_graphs: tuple[str, ...] = (
        "grid",
        "ring",
        "chain",
        "star",
        "uniform",
    )
    true_graphs: tuple[str, ...] = (
        "grid",
        "ring",
        "chain",
        "uniform",
    )
    mix_ratios: MixRatios | None = None
    mix_name: str | None = None
    eval_lengths: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1400)
    seq_len: int = 1400
    epsilon: float = 0.05
    alpha: float = 0.1
    seeds: tuple[int, ...] = tuple(range(16))
    output_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    @property
    def max_eval_length(self) -> int:
        return max(self.eval_lengths)


DEFAULT_CONFIG = ExperimentConfig()
