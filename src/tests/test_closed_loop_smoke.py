import tempfile
import unittest
from pathlib import Path

from src.experiments.calibrate_setpoints import main as calibrate_main
from src.experiments.run_experiment import main


class ClosedLoopSmokeTest(unittest.TestCase):
    def test_closed_loop_smoke_outputs_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            main([
                "--config", "src/config/default.yaml",
                "--scenario", "peak_demand",
                "--baseline", "fixed_signal_fixed_speed",
                "--controller", "stackelberg_mpc",
                "--T-total", "180",
                "--mpc-horizon-steps", "1",
                "--leader-candidate-count", "2",
                "--max-nash-iter", "2",
                "--freeway-horizon-beam-width", "1",
                "--freeway-ramp-candidate-limit", "1",
                "--freeway-vsl-candidate-limit", "1",
                "--output", str(out),
            ])
            attempt = out / "attempt_0"
            self.assertTrue((attempt / "config_used.yaml").exists())
            self.assertTrue((attempt / "metrics_summary.json").exists())
            self.assertTrue((attempt / "diagnostics.json").exists())
            self.assertTrue((attempt / "baseline" / "run_log.csv").exists())
            self.assertTrue((attempt / "proposed" / "control_timeseries.csv").exists())
            self.assertTrue((attempt / "proposed" / "state_timeseries.csv").exists())
            self.assertTrue((out / "report.md").exists())

    def test_setpoint_calibration_smoke_outputs_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "calibration"
            calibrate_main([
                "--config", "src/config/default.yaml",
                "--scenario", "peak_demand",
                "--baseline", "fixed_signal_fixed_speed",
                "--urban-scales", "0.75,1.0",
                "--T-total", "180",
                "--output", str(out),
            ])
            self.assertTrue((out / "mfd_points.csv").exists())
            self.assertTrue((out / "setpoint_calibration_summary.json").exists())
            self.assertTrue((out / "setpoint_calibration_report.md").exists())


if __name__ == "__main__":
    unittest.main()
