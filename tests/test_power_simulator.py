"""Unit tests for power subsystem simulator."""

import unittest
import numpy as np
from simulator.power import PowerSimulator, PowerTelemetry


class TestPowerSimulator(unittest.TestCase):
    """Test power subsystem simulator."""

    def setUp(self):
        self.sim = PowerSimulator(duration_hours=12)

    def test_simulator_initialization(self):
        """Test simulator initializes with correct dimensions."""
        self.assertEqual(len(self.sim.time), self.sim.num_samples)
        self.assertAlmostEqual(self.sim.time[0], 0)
        self.assertAlmostEqual(self.sim.time[-1], 12 * 3600, delta=1)

    def test_nominal_scenario(self):
        """Test nominal (healthy) scenario generates valid telemetry."""
        telemetry = self.sim.run_nominal()

        self.assertEqual(len(telemetry.solar_input), self.sim.num_samples)
        self.assertEqual(len(telemetry.battery_voltage), self.sim.num_samples)
        self.assertEqual(len(telemetry.battery_charge), self.sim.num_samples)
        self.assertEqual(len(telemetry.bus_voltage), self.sim.num_samples)

        self.assertTrue(np.all(telemetry.solar_input >= 0))
        self.assertTrue(np.all(telemetry.solar_input <= 550))

        self.assertTrue(np.all(telemetry.battery_charge >= 20))
        self.assertTrue(np.all(telemetry.battery_charge <= 100))

        self.assertTrue(np.all(telemetry.battery_voltage >= 20))
        self.assertTrue(np.all(telemetry.battery_voltage <= 30))

        self.assertTrue(np.all(telemetry.bus_voltage >= 9.5))
        self.assertTrue(np.all(telemetry.bus_voltage <= 13))

    def test_degraded_scenario(self):
        """Test degraded scenario with multiple faults."""
        telemetry = self.sim.run_degraded(
            solar_degradation_hour=3,
            solar_factor=0.6,
            battery_degradation_hour=4,
            battery_factor=0.75,
        )

        self.assertEqual(len(telemetry.solar_input), self.sim.num_samples)

        degrad_start_idx = int(3 * 3600 * self.sim.sampling_rate_hz)
        pre_degrad_solar = telemetry.solar_input[:degrad_start_idx].mean()
        post_degrad_solar = telemetry.solar_input[degrad_start_idx:].mean()

        self.assertLess(post_degrad_solar, pre_degrad_solar * 0.8)

    def test_solar_input_ranges(self):
        """Test solar input stays within physical bounds."""
        solar = self.sim.simulate_solar_input(
            base_power=500, eclipse_frequency_hours=1.5
        )
        self.assertTrue(np.all(solar >= 0))
        self.assertTrue(np.all(solar <= 550))

    def test_battery_dynamics_with_degradation(self):
        """Test battery charge degrades under efficiency loss."""
        solar_input = self.sim.simulate_solar_input(base_power=400)

        charge_nom, _ = self.sim.simulate_battery_dynamics(
            solar_input, efficiency_degradation_start_hour=None
        )

        charge_deg, _ = self.sim.simulate_battery_dynamics(
            solar_input, efficiency_degradation_start_hour=6, efficiency_factor=0.7
        )

        degrad_idx = int(6 * 3600 * self.sim.sampling_rate_hz)

        charge_diff = charge_nom[degrad_idx:].mean() - charge_deg[degrad_idx:].mean()
        self.assertGreater(charge_diff, 0, "Degraded charge should be lower")


if __name__ == "__main__":
    unittest.main()
