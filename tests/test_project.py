"""Behavioral and end-to-end checks for the published experiment."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from minisom_colors import ExperimentConfig, locate_color, train
from minisom_colors.artifacts import TrainingAnimation, load_weights, save_artifacts
from minisom_colors.cli import main

SMALL = ExperimentConfig(training_samples=30, test_samples=20, width=4, height=3,
                         iterations=50, sigma=1.0)


class ExperimentTests(unittest.TestCase):
    def test_same_seed_reproduces_the_map(self):
        a, b = train(SMALL), train(SMALL)
        np.testing.assert_array_equal(a.after, b.after)
        self.assertEqual(a.metrics["after"], b.metrics["after"])

    def test_evaluation_size_does_not_change_training(self):
        a = train(SMALL)
        b = train(replace(SMALL, test_samples=70))
        np.testing.assert_array_equal(a.colors, b.colors)
        np.testing.assert_array_equal(a.after, b.after)
        self.assertFalse(np.shares_memory(a.colors, a.test_colors))

    def test_recording_does_not_change_the_algorithm(self):
        frames = []
        observed = train(SMALL, frame_callback=lambda w, n: frames.append((w, n)),
                         frame_interval=17)
        plain = train(SMALL)
        np.testing.assert_array_equal(observed.after, plain.after)
        self.assertEqual([n for _, n in frames], [0, 17, 34, 50])
        np.testing.assert_array_equal(frames[-1][0], observed.after)
        self.assertFalse(np.shares_memory(frames[0][0], observed.after))

    def test_default_experiment_matches_reported_metrics(self):
        result = train()
        self.assertAlmostEqual(result.metrics["before"]["topographic_error"], .976)
        self.assertAlmostEqual(result.metrics["after"]["topographic_error"], .092)
        self.assertAlmostEqual(result.metrics["after"]["quantization_error"],
                               .09184821559659966, places=10)
        self.assertEqual(result.metrics["winner_row_column"], [15, 2])
        self.assertTrue(np.all((result.after >= 0) & (result.after <= 1)))

    def test_invalid_configuration_is_rejected(self):
        cases = [dict(iterations=0), dict(training_samples=-1), dict(width=1, height=1),
                 dict(sigma=float("nan")), dict(learning_rate=1.2), dict(seed=-1)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train(replace(SMALL, **changes))

    def test_nearest_color_on_a_known_map(self):
        weights = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]])
        match = locate_color(weights, [.9, .05, .03])
        self.assertEqual(match["winner_row_column"], [0, 0])
        self.assertEqual(match["winner_prototype_rgb"], [1.0, 0.0, 0.0])

    def test_invalid_queries_and_model_arrays_are_rejected(self):
        for query in [[1, 0], [1.1, 0, 0], [float("nan"), 0, 0]]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                locate_color(np.zeros((2, 2, 3)), query)
        for weights in [np.zeros((2, 3)), np.full((2, 2, 3), np.inf)]:
            with self.assertRaises(ValueError):
                locate_color(weights, [0, 0, 0])


class ArtifactTests(unittest.TestCase):
    def test_saved_weights_and_metrics_round_trip(self):
        result = train(replace(SMALL, training_samples=37))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = save_artifacts(result, directory)
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(p.is_file() and p.stat().st_size for p in paths))
            np.testing.assert_array_equal(load_weights(directory / "color_map.npz"),
                                          result.after)
            metrics = json.loads((directory / "results.json").read_text())
            self.assertEqual(metrics["after"], result.metrics["after"])

    def test_animation_records_initial_and_final_states(self):
        from PIL import Image
        animation = TrainingAnimation(pixels=80)
        train(SMALL, frame_callback=animation.capture, frame_interval=25)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "training.gif"
            animation.save(path)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 3)
                self.assertEqual(gif.info["loop"], 0)

    def test_model_without_trained_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.npz"
            np.savez(path, initial=np.zeros((2, 2, 3)))
            with self.assertRaisesRegex(ValueError, "trained weights"):
                load_weights(path)

    def test_static_export_removes_animation_from_a_previous_run(self):
        result = train(SMALL)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "training.gif").write_bytes(b"previous run")
            save_artifacts(result, directory)
            self.assertFalse((directory / "training.gif").exists())


class CommandLineTests(unittest.TestCase):
    def test_training_command_respects_options_and_exports_results(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            code = main(["train", "--training-samples", "30", "--test-samples", "20",
                         "--width", "4", "--height", "3", "--iterations", "50",
                         "--sigma", "1", "--output", temp, "--no-gif"])
            self.assertEqual(code, 0)
            result = json.loads((Path(temp) / "results.json").read_text())
            self.assertEqual(result["grid"], [4, 3])
            self.assertEqual(result["iterations"], 50)
            self.assertFalse((Path(temp) / "training.gif").exists())

    def test_locate_command_loads_the_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "map.npz"
            np.savez(path, after=np.array([[[1., 0., 0.], [0., 1., 0.]]]))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["locate", "--model", str(path),
                                       "--rgb", "0.9", "0", "0"]), 0)
            self.assertEqual(json.loads(output.getvalue())["winner_row_column"], [0, 0])

    def test_invalid_command_configuration_has_an_actionable_error(self):
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            main(["train", "--iterations", "0", "--no-gif"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("iterations must be a positive integer", error.getvalue())


if __name__ == "__main__":
    unittest.main()
