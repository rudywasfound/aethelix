"""Unit tests for causal reasoning and root cause ranking."""

import unittest
import numpy as np
from simulator.power import PowerSimulator
from causal_graph.graph_definition import CausalGraph, NodeType
from causal_graph.root_cause_ranking import RootCauseRanker


class TestCausalGraph(unittest.TestCase):
    """Test causal graph construction."""

    def setUp(self):
        self.graph = CausalGraph()

    def test_graph_initialization(self):
        """Test graph initializes with expected nodes."""
        self.assertGreater(len(self.graph.nodes), 0)
        self.assertGreater(len(self.graph.edges), 0)

    def test_node_types(self):
        """Test all expected node types exist."""
        root_causes = self.graph.get_root_causes()
        observables = self.graph.get_observables()

        self.assertGreater(len(root_causes), 0)
        self.assertGreater(len(observables), 0)

    def test_edges_connect_valid_nodes(self):
        """Test all edges connect nodes that exist."""
        for edge in self.graph.edges:
            self.assertIn(edge.source, self.graph.nodes)
            self.assertIn(edge.target, self.graph.nodes)

    def test_get_parents_and_children(self):
        """Test parent/child traversal."""
        if "battery_state" in self.graph.nodes:
            parents = self.graph.get_parents("battery_state")
            children = self.graph.get_children("battery_state")

            self.assertGreater(len(parents), 0, "battery_state should have parents")
            self.assertGreater(len(children), 0, "battery_state should have children")

    def test_paths_to_root(self):
        """Test path finding from observables to roots."""
        observable = "battery_voltage_measured"
        paths = self.graph.get_paths_to_root(observable)

        self.assertGreater(len(paths), 0)

        for path in paths:
            self.assertGreater(len(path), 1, "Paths should have at least 2 nodes")


class TestRootCauseRanker(unittest.TestCase):
    """Test root cause ranking algorithm."""

    def setUp(self):
        self.sim = PowerSimulator(duration_hours=12)
        self.graph = CausalGraph()
        self.ranker = RootCauseRanker(self.graph)

    def test_analyze_returns_hypotheses(self):
        """Test that analysis returns ranked hypotheses."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(
            solar_degradation_hour=3.0,
            battery_degradation_hour=4.0,
        )

        hypotheses = self.ranker.analyze(nominal, degraded)

        self.assertGreater(len(hypotheses), 0, "Should detect root causes")

    def test_hypotheses_are_ranked(self):
        """Test hypotheses are sorted by probability."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(
            solar_degradation_hour=3.0,
            battery_degradation_hour=4.0,
        )

        hypotheses = self.ranker.analyze(nominal, degraded)

        if len(hypotheses) > 1:
            for i in range(len(hypotheses) - 1):
                self.assertGreaterEqual(
                    hypotheses[i].probability,
                    hypotheses[i + 1].probability,
                    "Hypotheses should be sorted by probability",
                )

    def test_probabilities_sum_to_one(self):
        """Test that probabilities sum to 1.0."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(solar_degradation_hour=3.0)

        hypotheses = self.ranker.analyze(nominal, degraded)

        if len(hypotheses) > 0:
            total_prob = sum(h.probability for h in hypotheses)
            self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_solar_degradation_detected(self):
        """Test that solar degradation is detected and ranked."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(
            solar_degradation_hour=3.0,
            solar_factor=0.6,
            battery_degradation_hour=999.0,
        )

        hypotheses = self.ranker.analyze(nominal, degraded)

        hypothesis_names = [h.name for h in hypotheses]
        self.assertIn(
            "solar_degradation",
            hypothesis_names,
            "Solar degradation should be detected",
        )

        top_hypothesis = hypotheses[0]
        self.assertEqual(
            top_hypothesis.name,
            "solar_degradation",
            "Solar degradation should be top-ranked",
        )

    def test_battery_aging_detected(self):
        """Test that battery aging is detected and ranked."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(
            solar_degradation_hour=999.0,
            battery_degradation_hour=3.0,
            battery_factor=0.7,
        )

        hypotheses = self.ranker.analyze(nominal, degraded)

        hypothesis_names = [h.name for h in hypotheses]
        self.assertIn(
            "battery_aging",
            hypothesis_names,
            "Battery aging should be detected",
        )

    def test_hypotheses_have_evidence(self):
        """Test that hypotheses include supporting evidence."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(solar_degradation_hour=3.0)

        hypotheses = self.ranker.analyze(nominal, degraded)

        for hyp in hypotheses:
            self.assertGreater(
                len(hyp.evidence), 0, f"{hyp.name} should have evidence"
            )

    def test_confidence_is_valid(self):
        """Test that confidence scores are valid (0-1)."""
        nominal = self.sim.run_nominal()
        degraded = self.sim.run_degraded(solar_degradation_hour=3.0)

        hypotheses = self.ranker.analyze(nominal, degraded)

        for hyp in hypotheses:
            self.assertGreaterEqual(
                hyp.confidence, 0.0, "Confidence should be >= 0"
            )
            self.assertLessEqual(hyp.confidence, 1.0, "Confidence should be <= 1")


if __name__ == "__main__":
    unittest.main()
