"""Save actual data, model weights, figures and recorded training frames."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .experiment import ExperimentResult


class TrainingAnimation:
    def __init__(self, pixels=512):
        self.frames = []
        self.pixels = pixels
        self.font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 23)

    def capture(self, weights, completed):
        pixels = np.rint(np.clip(weights, 0, 1) * 255).astype(np.uint8)
        image = Image.new("RGB", (self.pixels + 48, self.pixels + 108), "white")
        image.paste(Image.fromarray(pixels).resize(
            (self.pixels, self.pixels), Image.Resampling.NEAREST), (24, 84))
        ImageDraw.Draw(image).text(
            (24, 24), f"MiniSom   {completed:,} updates", fill="#192C3B", font=self.font)
        self.frames.append(image)

    def save(self, output: Path):
        if len(self.frames) < 2:
            raise ValueError("An animation needs an initial and final frame")
        durations = [1200] + [130] * (len(self.frames) - 2) + [2200]
        self.frames[0].save(
            output, save_all=True, append_images=self.frames[1:],
            duration=durations, loop=0,
        )


def save_artifacts(result: ExperimentResult, output: Path, animation=None) -> list[Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    labels = ["Before training", f"After {result.config.iterations:,} updates"]
    for ax, weights, title in zip(axes, [result.before, result.after], labels):
        ax.imshow(weights, interpolation="nearest")
        ax.set_title(title, fontsize=21, pad=16)
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.04, wspace=.09)
    fig.savefig(output / "before_after.png", dpi=160, facecolor="white")
    plt.close(fig)
    for name, weights in [("map_before", result.before), ("map_after", result.after)]:
        plt.imsave(output / f"{name}.png", weights)

    # Unused positions in a partial last row stay transparent.
    count = len(result.colors)
    columns = min(50, count)
    rows = (count + columns - 1) // columns
    grid = np.zeros((rows * columns, 4))
    grid[:count, :3] = result.colors
    grid[:count, 3] = 1
    plt.imsave(output / "input_colors.png", grid.reshape(rows, columns, 4))
    np.savez_compressed(
        output / "color_map.npz", before=result.before, after=result.after,
        config_json=json.dumps(result.metrics["config"]),
    )
    (output / "results.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")
    paths = [output / name for name in ["before_after.png", "map_before.png",
             "map_after.png", "input_colors.png", "color_map.npz", "results.json"]]
    if animation is not None:
        animation.save(output / "training.gif")
        paths.append(output / "training.gif")
    elif (output / "training.gif").exists():
        # Keep a previous run's animation from being mistaken for this run.
        (output / "training.gif").unlink()
    return paths


def load_weights(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "after" not in archive:
            raise ValueError("The model archive does not contain trained weights ('after')")
        return archive["after"].copy()
