"""Unit tests for thermal subsystem simulator."""

import unittest
import numpy as np
from simulator.power import PowerSimulator
from simulator.thermal import ThermalSimulator


class TestThermalSimulator(unittest.TestCase):
    """Test thermal subsystem simulator."""

    def setUp(self):
        self.power_sim = PowerSimulator(duration_hours=12)
        self.thermal_sim = ThermalSimulator(duration_hours=12)
        
        self.power_nominal = self.power_sim.run_nominal()

    def test_thermal_simulator_initialization(self):
        """Test thermal simulator initializes with correct dimensions."""
        self.assertEqual(len(self.thermal_sim.time), self.thermal_sim.num_samples)
        self.assertAlmostEqual(self.thermal_sim.time[0], 0)
        self.assertAlmostEqual(self.thermal_sim.time[-1], 12 * 3600, delta=1)

    def test_nominal_thermal_scenario(self):
        """Test nominal (healthy) thermal scenario generates valid telemetry."""
        telemetry = self.thermal_sim.run_nominal(
            self.power_nominal.solar_input,
            self.power_nominal.battery_charge,
            self.power_nominal.battery_voltage,
        )

        self.assertEqual(len(telemetry.battery_temp), self.thermal_sim.num_samples)
        self.assertEqual(len(telemetry.solar_panel_temp), self.thermal_sim.num_samples)
        self.assertEqual(len(telemetry.payload_temp), self.thermal_sim.num_samples)
        self.assertEqual(len(telemetry.bus_current), self.thermal_sim.num_samples)

        self.assertTrue(np.all(telemetry.battery_temp >= 20))
        self.assertTrue(np.all(telemetry.battery_temp <= 65))

        self.assertTrue(np.all(telemetry.solar_panel_temp >= 0))
        self.assertTrue(np.all(telemetry.solar_panel_temp <= 70))

        self.assertTrue(np.all(telemetry.payload_temp >= 20))
        self.assertTrue(np.all(telemetry.payload_temp <= 55))

        self.assertTrue(np.all(telemetry.bus_current >= 0))
        self.assertTrue(np.all(telemetry.bus_current <= 60))

    def test_degraded_thermal_scenario(self):
        """Test degraded thermal scenario with faults."""
        telemetry = self.thermal_sim.run_degraded(
            self.power_nominal.solar_input,
            self.power_nominal.battery_charge,
            self.power_nominal.battery_voltage,
            panel_degradation_hour=3.0,
            battery_cooling_hour=4.0,
        )

        self.assertEqual(len(telemetry.battery_temp), self.thermal_sim.num_samples)

    def test_solar_panel_temp_oscillates(self):
        """Test solar panel temperature varies with eclipse cycles."""
        temp = self.thermal_sim.simulate_solar_panel_temp(eclipse_frequency_hours=1.5)

        temp_range = np.max(temp) - np.min(temp)
        self.assertGreater(temp_range, 10, "Panel temperature should vary with eclipse cycle")

    def test_battery_temp_increases_with_stress(self):
        """Test battery temperature responds to charge/discharge stress."""
        charge_profile = np.linspace(80, 20, self.thermal_sim.num_samples)
        solar_high = np.full(self.thermal_sim.num_samples, 400.0)

        temp = self.thermal_sim.simulate_battery_temp(
            solar_high, charge_profile, degradation_start_hour=None
        )

        early_temp = np.mean(temp[:len(temp)//4])
        late_temp = np.mean(temp[3*len(temp)//4:])

        self.assertLess(early_temp, late_temp, "Battery should heat up with discharge stress")

    def test_panel_temp_degradation_visible(self):
        """Test that panel temperature degradation is detectable."""
        temp_healthy = self.thermal_sim.simulate_solar_panel_temp(
            degradation_start_hour=None
        )

        temp_degraded = self.thermal_sim.simulate_solar_panel_temp(
            degradation_start_hour=3.0,
            degradation_drift_rate=0.5,
        )

        degrad_idx = int(3 * 3600 * self.thermal_sim.sampling_rate_hz)

        pre_degrad_healthy = np.mean(temp_healthy[:degrad_idx])
        post_degrad_degraded = np.mean(temp_degraded[degrad_idx:])

        self.assertLess(
            pre_degrad_healthy,
            post_degrad_degraded,
            "Degraded panel should run hotter",
        )

    def test_battery_cooling_failure_visible(self):
        """Test that heatsink failure causes temperature rise."""
        charge_profile = np.linspace(80, 30, self.thermal_sim.num_samples)
        solar_input = np.full(self.thermal_sim.num_samples, 300.0)

        temp_healthy = self.thermal_sim.simulate_battery_temp(
            solar_input, charge_profile, degradation_start_hour=None
        )

        temp_degraded = self.thermal_sim.simulate_battery_temp(
            solar_input,
            charge_profile,
            degradation_start_hour=3.0,
            degradation_factor=0.3,
        )

        degrad_idx = int(3 * 3600 * self.thermal_sim.sampling_rate_hz)

        pre_degrad_healthy = np.mean(temp_healthy[:degrad_idx])
        post_degrad_healthy = np.mean(temp_healthy[degrad_idx:])
        post_degrad_degraded = np.mean(temp_degraded[degrad_idx:])

        self.assertGreater(
            post_degrad_degraded,
            post_degrad_healthy,
            "Failed cooling should increase temperature",
        )

    def test_payload_temp_bounded(self):
        """Test payload temperature stays within physical bounds."""
        voltage = np.linspace(22, 28, self.thermal_sim.num_samples)

        temp = self.thermal_sim.simulate_payload_temp(voltage)

        self.assertTrue(np.all(temp >= 20), "Payload temp should be above ambient")
        self.assertTrue(np.all(temp <= 55), "Payload temp should be below max safe")

    def test_bus_current_increases_with_stress(self):
        """Test bus current reflects battery stress."""
        charge_high = np.full(self.thermal_sim.num_samples, 90.0)
        charge_low = np.full(self.thermal_sim.num_samples, 25.0)

        voltage = np.full(self.thermal_sim.num_samples, 26.0)

        current_high = self.thermal_sim.simulate_bus_current(charge_high, voltage)
        current_low = self.thermal_sim.simulate_bus_current(charge_low, voltage)

        mean_high = np.mean(current_high)
        mean_low = np.mean(current_low)

        self.assertLess(
            mean_high,
            mean_low,
            "Low battery charge should increase regulation current",
        )


class TestThermalPowerIntegration(unittest.TestCase):
    """Test integration between power and thermal simulators."""

    def test_combined_scenario(self):
        """Test running power and thermal simulations together."""
        power_sim = PowerSimulator(duration_hours=6)
        thermal_sim = ThermalSimulator(duration_hours=6)

        power_nominal = power_sim.run_nominal()
        power_degraded = power_sim.run_degraded(solar_degradation_hour=2.0)

        thermal_nominal = thermal_sim.run_nominal(
            power_nominal.solar_input,
            power_nominal.battery_charge,
            power_nominal.battery_voltage,
        )

        thermal_degraded = thermal_sim.run_degraded(
            power_degraded.solar_input,
            power_degraded.battery_charge,
            power_degraded.battery_voltage,
            battery_cooling_hour=2.5,
        )

        self.assertEqual(len(thermal_nominal.battery_temp), thermal_sim.num_samples)
        self.assertEqual(len(thermal_degraded.battery_temp), thermal_sim.num_samples)

        self.assertGreater(
            np.mean(thermal_degraded.battery_temp),
            np.mean(thermal_nominal.battery_temp),
        )


if __name__ == "__main__":
    unittest.main()
