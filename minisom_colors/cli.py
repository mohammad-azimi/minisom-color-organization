"""Command line entry points for the experiment and saved-model queries."""

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, locate_color, train


def build_parser():
    parser = argparse.ArgumentParser(description="Organize RGB colors with MiniSom")
    commands = parser.add_subparsers(dest="command")
    training = commands.add_parser("train", help="Train, evaluate and save a color map")
    for name, default in [("training-samples", 1000), ("test-samples", 500),
                          ("width", 16), ("height", 16), ("iterations", 20000), ("seed", 42)]:
        training.add_argument("--" + name, type=int, default=default)
    training.add_argument("--sigma", type=float, default=2.0)
    training.add_argument("--learning-rate", type=float, default=0.5)
    training.add_argument("--output", type=Path, default=Path("outputs"))
    training.add_argument("--query", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                          metavar=("R", "G", "B"))
    training.add_argument("--no-gif", action="store_true", help="Skip animation recording")
    training.add_argument("--frame-interval", type=int, default=500)
    query = commands.add_parser("locate", help="Locate a color using saved model weights")
    query.add_argument("--model", type=Path, default=Path("examples/color_map.npz"))
    query.add_argument("--rgb", nargs=3, required=True, type=float, metavar=("R", "G", "B"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["train"])
    try:
        if args.command == "locate":
            from .artifacts import load_weights
            print(json.dumps(locate_color(load_weights(args.model), args.rgb), indent=2))
            return 0
        from .artifacts import TrainingAnimation, save_artifacts
        config = ExperimentConfig(**{
            name: getattr(args, name) for name in ExperimentConfig.__dataclass_fields__
        })
        config.validate()
        animation = None if args.no_gif else TrainingAnimation()
        result = train(config, frame_callback=animation.capture if animation else None,
                       frame_interval=args.frame_interval, query=args.query)
        paths = save_artifacts(result, args.output, animation)
        print("Training finished")
        for name, label in [("topographic_error", "Topographic error"),
                            ("quantization_error", "Quantization error")]:
            print(f"{label}: {result.metrics['before'][name]:.6f} -> "
                  f"{result.metrics['after'][name]:.6f}")
        print("Query location (row, column):", tuple(result.metrics["winner_row_column"]))
        print(f"Saved {len(paths)} files to {args.output.resolve()}")
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
