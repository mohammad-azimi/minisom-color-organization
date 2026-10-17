"""Training and evaluation, independent of plotting or the command line."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import math
import platform
from typing import Callable

import numpy as np
from minisom import MiniSom

FrameCallback = Callable[[np.ndarray, int], None]


@dataclass(frozen=True)
class ExperimentConfig:
    training_samples: int = 1000
    test_samples: int = 500
    width: int = 16
    height: int = 16
    iterations: int = 20000
    sigma: float = 2.0
    learning_rate: float = 0.5
    seed: int = 42

    def validate(self) -> None:
        for name in ("training_samples", "test_samples", "width", "height", "iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width * self.height < 2:
            raise ValueError("The map needs at least two cells for topographic error")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and greater than zero")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    colors: np.ndarray
    test_colors: np.ndarray
    before: np.ndarray
    after: np.ndarray
    metrics: dict


class _ObservedMiniSom(MiniSom):
    """Observe MiniSom's public update method without changing its algorithm."""

    def __init__(self, *args, frame_callback=None, frame_interval=500, **kwargs):
        self._frame_callback = frame_callback
        self._frame_interval = frame_interval
        super().__init__(*args, **kwargs)

    def update(self, x, win, t, max_iteration):
        super().update(x, win, t, max_iteration)
        completed = t + 1
        if self._frame_callback and (
            completed % self._frame_interval == 0 or completed == max_iteration
        ):
            self._frame_callback(self.get_weights().copy(), completed)


def locate_color(weights: np.ndarray, rgb) -> dict:
    """Return the nearest prototype using Euclidean RGB distance."""
    weights = np.asarray(weights, dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    if weights.ndim != 3 or weights.shape[2] != 3 or 0 in weights.shape:
        raise ValueError("Model weights must have shape (rows, columns, 3)")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Model weights must contain finite RGB values in [0, 1]")
    if rgb.shape != (3,) or not np.isfinite(rgb).all():
        raise ValueError("A query must contain three finite RGB values")
    if np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("Query RGB values must be in [0, 1]")
    distances = np.linalg.norm(weights - rgb, axis=2)
    row, column = (int(v) for v in np.unravel_index(distances.argmin(), distances.shape))
    return {
        "query_rgb": rgb.tolist(),
        "winner_row_column": [row, column],
        "winner_prototype_rgb": weights[row, column].tolist(),
        "distance": float(distances[row, column]),
    }


def train(config: ExperimentConfig | None = None, *, frame_callback=None,
          frame_interval: int = 500, query=(1.0, 0.0, 0.0)) -> ExperimentResult:
    """Use the same held-out colors to evaluate a SOM before and after training."""
    config = config or ExperimentConfig()
    config.validate()
    if type(frame_interval) is not int or frame_interval < 1:
        raise ValueError("frame_interval must be a positive integer")
    # Validate queries before spending time on training.
    locate_color(np.zeros((1, 1, 3)), query)
    rng = np.random.default_rng(config.seed)
    colors = rng.random((config.training_samples, 3))
    test_colors = rng.random((config.test_samples, 3))
    som = _ObservedMiniSom(
        config.width, config.height, 3, sigma=config.sigma,
        learning_rate=config.learning_rate, random_seed=config.seed,
        frame_callback=frame_callback, frame_interval=frame_interval,
    )
    som.random_weights_init(colors)
    before = som.get_weights().copy()
    if frame_callback:
        frame_callback(before.copy(), 0)

    def evaluate():
        return {
            "topographic_error": float(som.topographic_error(test_colors)),
            "quantization_error": float(som.quantization_error(test_colors)),
        }

    before_metrics = evaluate()
    som.train_random(colors, config.iterations)
    after = som.get_weights().copy()
    after_metrics = evaluate()
    metrics = {
        "schema_version": 1,
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(), "minisom": version("MiniSom"),
            "numpy": np.__version__, "matplotlib": version("matplotlib"),
            "pillow": version("Pillow"),
        },
        "training_samples": config.training_samples, "test_samples": config.test_samples,
        "grid": [config.width, config.height], "input_features": 3,
        "iterations": config.iterations, "seed": config.seed,
        "sigma": config.sigma, "learning_rate": config.learning_rate,
        "before": before_metrics, "after": after_metrics,
        **locate_color(after, query),
        "data_description": "Synthetic uniform RGB values in [0,1], no labels",
        "evaluation": "Same held-out RGB vectors before and after training",
    }
    return ExperimentResult(config, colors, test_colors, before, after, metrics)
