"""
Causal graph definition for satellite power and thermal subsystems.

This module encodes engineering domain knowledge as a directed acyclic graph (DAG).
The graph represents how failures propagate through satellites:

Root Causes (faults) --> Intermediate Effects --> Observable Deviations

Example path:
solar_degradation --> reduced solar_input --> battery can't charge properly --> 
lower battery_charge measured in telemetry

Why a causal graph:
1. Explicit representation of failure mechanisms (transparent to domain experts)
2. Enables path tracing from observables back to root causes
3. Supports multi-fault diagnosis (traces multiple causes simultaneously)
4. Domain experts (ISRO engineers) can validate and refine the structure
5. Handles confounding effects (one fault causing secondary deviations)

The graph structure:
- 7 ROOT CAUSES (primary faults we want to identify)
- 8 INTERMEDIATE nodes (effects that propagate between subsystems)
- 8 OBSERVABLE nodes (measured telemetry we can see)
- 29 directed edges with weights and mechanisms

This encoding enables Bayesian causal inference to rank hypotheses about
which root causes best explain observed deviations in telemetry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from enum import Enum

try:
    from aethelix_core import PyCausalGraph
    RUST_CORE_AVAILABLE = True
except ImportError:
    RUST_CORE_AVAILABLE = False
    PyCausalGraph = None


class NodeType(Enum):
    """
    Types of nodes in causal graph.
    
    Each type represents a different role in the failure propagation chain:
    - ROOT_CAUSE: Primary faults (what we want to diagnose)
    - INTERMEDIATE: Effects propagating through subsystems
    - OBSERVABLE: Measured telemetry we can actually see
    """
    
    ROOT_CAUSE = "root_cause"  # Primary fault sources (the diagnosis targets)
    INTERMEDIATE = "intermediate"  # Propagation nodes (unobservable state)
    OBSERVABLE = "observable"  # Measured telemetry (what we observe)


@dataclass
class Node:
    """
    A node in the causal graph.
    
    Each node represents some aspect of the satellite:
    - A root cause (fault) that needs diagnosis
    - An intermediate physical effect (unobservable but inferred)
    - An observable measurement from telemetry
    
    The description and degradation_modes help domain experts understand
    what each node represents and how it can fail.
    """
    
    name: str                           # Unique identifier
    node_type: NodeType                 # Is this a cause, intermediate, or observable?
    description: str                    # Natural language explanation for operators
    degradation_modes: List[str] = field(default_factory=list)  # How can this node fail?


@dataclass
class Edge:
    """
    A directed causal edge (parent → child).
    
    An edge from A to B means "failures in A cause effects in B".
    The weight quantifies strength: 0.9 means strong causal effect,
    0.5 means weak effect.
    
    The mechanism is crucial for explainability: when we rank a hypothesis,
    we can show the user "Solar degradation likely because: reduced input power
    cannot recharge battery (mechanism) -> battery charge drops (observable)".
    """
    
    source: str             # Source node name (cause)
    target: str             # Target node name (effect)
    weight: float = 1.0     # Strength of causal relationship (0-1, higher = stronger)
    mechanism: str = ""     # How source affects target (for explanation)


class CausalGraph:
    """
    DAG representing causal relationships in power and thermal subsystems.
    Encodes engineering knowledge for Bayesian root cause inference.
    """

    def __init__(self):
        """Initialize graph and subsystems."""
        
        self.nodes: Dict[str, Node] = {}  # name -> Node object
        self.edges: List[Edge] = []        # List of causal edges
        
        # High-performance Rust backend for complex graph operations
        if RUST_CORE_AVAILABLE:
            self.rust_graph = PyCausalGraph()
        else:
            self.rust_graph = None
        
        # Build the complete graph structure
        self._build_power_subsystem_graph()
        self._build_adcs_subsystem_graph()
        self._build_comms_subsystem_graph()
        self._build_obc_subsystem_graph()
        self._build_propulsion_subsystem_graph()
        self._build_cross_subsystem_coupling()

    def _build_power_subsystem_graph(self):
        """Build power/thermal graph layers: faults, effects, and telemetry."""

        # Root Causes

        # These are the faults we want to diagnose. Each represents a failure mode.
        
        # Power subsystem root causes
        self.add_node(
            "solar_degradation",
            NodeType.ROOT_CAUSE,
            "Solar panel efficiency loss or shadowing",
            degradation_modes=["panel_aging", "dust_accumulation", "partial_shadowing"],
        )

        
        self.add_node(
            "battery_aging",
            NodeType.ROOT_CAUSE,
            "Battery cell degradation and capacity loss",
            degradation_modes=["cell_aging", "internal_resistance_rise"],
        )

        
        self.add_node(
            "battery_thermal",
            NodeType.ROOT_CAUSE,
            "Excessive battery temperature stress",
            degradation_modes=["thermal_runaway_risk", "efficiency_loss"],
        )

        
        self.add_node(
            "sensor_bias",
            NodeType.ROOT_CAUSE,
            "Measurement bias or sensor drift",
            degradation_modes=["calibration_drift", "electronic_aging"],
        )


        # Thermal subsystem root causes
        self.add_node(
            "panel_insulation_degradation",
            NodeType.ROOT_CAUSE,
            "Solar panel insulation or radiator fouling",
            degradation_modes=["insulation_loss", "radiator_fouling"],
        )


        self.add_node(
            "battery_heatsink_failure",
            NodeType.ROOT_CAUSE,
            "Battery thermal management system failure",
            degradation_modes=["heatsink_blockage", "coolant_loss"],
        )


        self.add_node(
            "payload_radiator_degradation",
            NodeType.ROOT_CAUSE,
            "Payload electronics radiator degradation",
            degradation_modes=["radiator_coating_loss", "micrometeorite_damage"],
        )


        self.add_node(
            "pcdu_regulator_failure",
            NodeType.ROOT_CAUSE,
            "Regulated Power Bus or PCDU Regulator failure",
            degradation_modes=["regulator_short", "pcdu_controller_fault"],
        )


        # Intermediate Nodes


        # Power subsystem intermediates
        self.add_node(
            "solar_input",
            NodeType.INTERMEDIATE,
            "Available solar power from panels",
        )

        self.add_node(
            "battery_efficiency",
            NodeType.INTERMEDIATE,
            "Battery charge/discharge efficiency",
        )

        self.add_node(
            "battery_state",
            NodeType.INTERMEDIATE,
            "Battery charge capacity and health",
        )

        self.add_node(
            "bus_regulation",
            NodeType.INTERMEDIATE,
            "Bus voltage regulation quality",
        )

        # Thermal subsystem intermediates
        self.add_node(
            "solar_panel_temp",
            NodeType.INTERMEDIATE,
            "Solar panel temperature",
        )

        self.add_node(
            "battery_temp",
            NodeType.INTERMEDIATE,
            "Battery cell temperature",
        )

        self.add_node(
            "payload_temp",
            NodeType.INTERMEDIATE,
            "Payload electronics temperature",
        )

        self.add_node(
            "thermal_stress",
            NodeType.INTERMEDIATE,
            "Overall system thermal stress level",
        )

        # Observable Nodes


        # Power observables
        self.add_node(
            "solar_input_measured",
            NodeType.OBSERVABLE,
            "Measured solar input power",
        )

        self.add_node(
            "battery_voltage_measured",
            NodeType.OBSERVABLE,
            "Measured battery voltage",
        )

        self.add_node(
            "battery_charge_measured",
            NodeType.OBSERVABLE,
            "Measured battery charge state percentage",
        )

        self.add_node(
            "bus_voltage_measured",
            NodeType.OBSERVABLE,
            "Measured bus output voltage",
        )

        # Thermal observables
        self.add_node(
            "solar_panel_temp_measured",
            NodeType.OBSERVABLE,
            "Measured solar panel temperature",
        )

        self.add_node(
            "battery_temp_measured",
            NodeType.OBSERVABLE,
            "Measured battery temperature",
        )

        self.add_node(
            "payload_temp_measured",
            NodeType.OBSERVABLE,
            "Measured payload temperature",
        )

        self.add_node(
            "bus_current_measured",
            NodeType.OBSERVABLE,
            "Measured bus current (power dissipation proxy)",
        )

        # Power Subsystem Edges


        # Solar degradation directly affects available solar input
        self.add_edge(
            "solar_degradation",
            "solar_input",
            weight=0.95,  # Strong effect (degradation directly reduces output)
            mechanism="Reduced panel output due to physical degradation or shadowing",
        )

        # Battery aging reduces charging efficiency
        self.add_edge(
            "battery_aging",
            "battery_efficiency",
            weight=0.9,  # Increased to emphasize chemical degradation
            mechanism="Increased internal resistance reduces charge/discharge efficiency",
        )

        # Battery thermal stress reduces efficiency (temperature effects on electrochemistry)
        # Weight reduced here to shift primary observability to the temperature path
        self.add_edge(
            "battery_thermal",
            "battery_efficiency",
            weight=0.65,  
            mechanism="High temperature degrades battery electrochemistry and increases losses",
        )

        # New: Thermal signature for battery thermal stress
        self.add_edge(
            "battery_thermal",
            "battery_temp",
            weight=0.75,
            mechanism="Internal battery thermal stress manifests as temperature rise",
        )

        # Reduced solar input means battery can't recharge properly
        self.add_edge(
            "solar_input",
            "battery_state",
            weight=0.9,  # Strong effect
            mechanism="Reduced input power cannot recharge battery to nominal capacity",
        )

        # Lower efficiency means less power stored per unit input
        self.add_edge(
            "battery_efficiency",
            "battery_state",
            weight=0.85,  # Strong effect
            mechanism="Lower efficiency means less power actually stored per unit of solar input",
        )

        # Degraded battery makes voltage regulation harder
        self.add_edge(
            "battery_state",
            "bus_regulation",
            weight=0.8,
            mechanism="Degraded battery supply makes regulation harder and less stable",
        )

        # Solar input is directly measured
        self.add_edge(
            "solar_input",
            "solar_input_measured",
            weight=1.0,  # Nearly perfect measurement of physical quantity
            mechanism="Direct measurement of solar power via sensor",
        )

        # Battery state is reflected in measured voltage
        self.add_edge(
            "battery_state",
            "battery_voltage_measured",
            weight=0.95,  # Strong correlation (voltage sags with low charge)
            mechanism="Battery voltage reflects state of charge via electrochemical potential",
        )

        # Efficiency degradation causes voltage droop
        self.add_edge(
            "battery_efficiency",
            "battery_voltage_measured",
            weight=0.7,  # Moderate effect (efficiency affects voltage under load)
            mechanism="Efficiency degradation causes voltage droop due to increased internal resistance",
        )

        # Charge capacity is directly measured
        self.add_edge(
            "battery_state",
            "battery_charge_measured",
            weight=0.9,  # Strong measurement of physical quantity
            mechanism="Charge sensor reports actual state of charge of battery",
        )

        # Bus regulation quality affects measured bus voltage
        self.add_edge(
            "bus_regulation",
            "bus_voltage_measured",
            weight=0.95,  # Strong effect (regulator directly controls output)
            mechanism="Bus voltage sensor directly measures regulator output",
        )

        # Battery state affects available regulated power
        self.add_edge(
            "battery_state",
            "bus_voltage_measured",
            weight=0.75,  # Moderate effect (battery is source of regulated power)
            mechanism="Battery state affects available power for regulation",
        )

        # PCDU failure directly collapses bus regulation
        self.add_edge(
            "pcdu_regulator_failure",
            "bus_regulation",
            weight=0.98,  # Critical path
            mechanism="PCDU regulator failure directly collapses regulated voltage levels",
        )

        # PCDU failure affects bus current draw proxy
        self.add_edge(
            "pcdu_regulator_failure",
            "bus_current_measured",
            weight=0.8,
            mechanism="Failed regulator cannot sustain load current, dropping observed draw to zero",
        )

        # Sensor bias adds error to voltage measurements
        self.add_edge(
            "sensor_bias",
            "battery_voltage_measured",
            weight=0.5,  # Weak but consistent effect (bias is constant or slow-varying)
            mechanism="Sensor drift and calibration error add bias to voltage readings",
        )

        # Sensor bias affects charge estimation
        self.add_edge(
            "sensor_bias",
            "battery_charge_measured",
            weight=0.5,  # Weak effect (charge estimation also depends on other factors)
            mechanism="Sensor drift affects charge state estimation algorithms",
        )

        # Thermal Subsystem Edges


        # Solar input affects panel temperature (more sun = more heating)
        self.add_edge(
            "solar_input",
            "solar_panel_temp",
            weight=0.85,  # Strong effect (solar radiation is primary heat source)
            mechanism="Increased solar radiation heats panel (albedo and thermal effects)",
        )

        # Regulated power enables payload operation and heat generation
        self.add_edge(
            "bus_regulation",
            "payload_temp",
            weight=0.7,  # Moderate effect (heat from payload electronics)
            mechanism="Available regulated power enables payload operation, generating heat",
        )

        # Insulation degradation prevents cooling, raising panel temperature
        self.add_edge(
            "panel_insulation_degradation",
            "solar_panel_temp",
            weight=0.9,  # Strong effect (insulation is primary temperature control)
            mechanism="Poor insulation/radiator coating prevents radiative cooling to space",
        )

        # Heatsink failure raises battery temperature
        self.add_edge(
            "battery_heatsink_failure",
            "battery_temp",
            weight=0.95,  # Very strong effect (heatsink is primary cooling path)
            mechanism="Failed heatsink eliminates primary cooling path for battery heat dissipation",
        )

        # Radiator degradation prevents payload cooling
        self.add_edge(
            "payload_radiator_degradation",
            "payload_temp",
            weight=0.9,  # Strong effect (radiator is primary cooling)
            mechanism="Degraded radiator reduces heat dissipation to space",
        )

        # Battery temperature contributes to overall thermal stress
        self.add_edge(
            "battery_temp",
            "thermal_stress",
            weight=0.7,  # Significant contributor
            mechanism="High battery temperature is critical thermal stress indicator (risk of runaway)",
        )

        # Payload temperature contributes to thermal stress
        self.add_edge(
            "payload_temp",
            "thermal_stress",
            weight=0.6,  # Moderate contributor
            mechanism="High payload temperature increases mission risk (reduced margins)",
        )

        # Panel temperature contributes to thermal stress
        self.add_edge(
            "solar_panel_temp",
            "thermal_stress",
            weight=0.5,  # Lower priority contributor
            mechanism="High panel temperature indicates reduced thermal margin",
        )

        # Power-Thermal Coupling


        # High battery temperature reduces efficiency (feedback loop)
        self.add_edge(
            "battery_temp",
            "battery_efficiency",
            weight=0.7,  # Moderate feedback effect
            mechanism="Elevated temperature increases internal resistance and electrochemical losses",
        )

        # Thermal System Measurement Edges


        # Panel temperature is directly measured
        self.add_edge(
            "solar_panel_temp",
            "solar_panel_temp_measured",
            weight=0.98,  # Nearly perfect measurement
            mechanism="Direct temperature sensor measurement via thermistor",
        )

        # Battery temperature is directly measured
        self.add_edge(
            "battery_temp",
            "battery_temp_measured",
            weight=0.95,  # High fidelity measurement
            mechanism="Battery thermistor directly measures cell temperature",
        )

        # Payload temperature is directly measured
        self.add_edge(
            "payload_temp",
            "payload_temp_measured",
            weight=0.96,  # High fidelity measurement
            mechanism="Payload thermal sensor provides local temperature measurement",
        )

        # Battery state affects current draw (regulation effort)
        self.add_edge(
            "battery_state",
            "bus_current_measured",
            weight=0.8,  # Moderate effect
            mechanism="Low battery state increases regulation effort and current draw",
        )

        # Reduced efficiency requires higher current
        self.add_edge(
            "battery_efficiency",
            "bus_current_measured",
            weight=0.7,  # Moderate effect
            mechanism="Reduced efficiency requires higher current to deliver same power",
        )

    def _build_adcs_subsystem_graph(self):
        """
        Build ADCS (Attitude Determination and Control System) causal structure.
        
        WHY THIS MATTERS OPERATIONALLY:
        ADCS faults are the #1 cause of mission loss for small satellites.
        A reaction wheel failure doesn't just stop rotation; it creates
        induced jitter and thermal stress, impacting payload data quality.
        """

        # ========== ROOT CAUSES: ADCS ==========
        # ECSS-FM-AOCS-01: Reaction wheel bearing friction
        self.add_node(
            "wheel_friction",
            NodeType.ROOT_CAUSE,
            "Increased friction in reaction wheel bearings",
            degradation_modes=["bearing_wear", "lubricant_degradation"],
        )
        
        # ECSS-FM-AOCS-02: Gyroscope calibration drift
        self.add_node(
            "gyro_drift",
            NodeType.ROOT_CAUSE,
            "Uncompensated drift in gyroscope bias",
            degradation_modes=["thermal_drift", "calibration_loss"],
        )

        # ECSS-FM-AOCS-03: Magnetorquer electronic fault
        self.add_node(
            "magnetorquer_anomaly",
            NodeType.ROOT_CAUSE,
            "Electronic fault in BCT or magnetorquer coils",
            degradation_modes=["coil_short", "driver_fault"],
        )

        # ========== INTERMEDIATES: ADCS ==========
        self.add_node(
            "pointing_accuracy",
            NodeType.INTERMEDIATE,
            "Satellite attitude pointing precision",
        )

        self.add_node(
            "control_effort",
            NodeType.INTERMEDIATE,
            "Magnetic/Momentum control effort level",
        )

        # ADCS Observables

        self.add_node(
            "pointing_error_measured",
            NodeType.OBSERVABLE,
            "Measured pointing deviation (arcsec)",
        )

        self.add_node(
            "wheel_speed_measured",
            NodeType.OBSERVABLE,
            "Measured reaction wheel rotational speed (RPM)",
        )

        self.add_node(
            "wheel_current_measured",
            NodeType.OBSERVABLE,
            "Measured reaction wheel motor current (A)",
        )

        self.add_node(
            "gyro_bias_observed",
            NodeType.OBSERVABLE,
            "Estimated gyroscope bias from Kalman Filter",
        )

        # ADCS Edges

        # Friction increases current draw and reduces pointing stability
        self.add_edge("wheel_friction", "wheel_current_measured", weight=0.9, mechanism="Motor must work harder to overcome bearing friction")
        self.add_edge("wheel_friction", "pointing_accuracy", weight=0.6, mechanism="Induced jitter from bearing vibration")
        
        # Gyro drift causes fake errors that controller tries to fix
        self.add_edge("gyro_drift", "gyro_bias_observed", weight=0.95, mechanism="Direct estimation of bias by flight software")
        self.add_edge("gyro_drift", "pointing_accuracy", weight=0.8, mechanism="Controller corrects for fake bias, inducing real pointing error")

        # Magnetorquer failure prevents desaturation
        self.add_edge("magnetorquer_anomaly", "control_effort", weight=0.8, mechanism="Loss of magnetic desaturation capability")
        self.add_edge("control_effort", "wheel_speed_measured", weight=0.9, mechanism="Saturated momentum must be stored in wheels")

        # Connection to measurement
        self.add_edge("pointing_accuracy", "pointing_error_measured", weight=1.0, mechanism="Telemetry reports actual deviation")

    def _build_comms_subsystem_graph(self):
        """
        Build Communications subsystem causal structure.
        
        WHY THIS MATTERS OPERATIONALLY:
        A 'silent satellite' mode is the ultimate failure. Identifying HPA
        degradation before total loss allows for adaptive modulation switching.
        """

        # Comms Root Causes

        # ECSS-FM-COM-01: High Power Amplifier degradation
        self.add_node(
            "transponder_fault",
            NodeType.ROOT_CAUSE,
            "HPA or SSPA efficiency loss / degradation",
            degradation_modes=["semiconductor_aging", "thermal_stress"],
        )

        # ECSS-FM-COM-02: Antenna pointing misalignment
        self.add_node(
            "antenna_pointing_error",
            NodeType.ROOT_CAUSE,
            "Mechanical antenna pointing or feed misalignment",
            degradation_modes=["gimble_stuck", "thermal_distortion"],
        )

        # ECSS-FM-COM-03: Signal interference
        self.add_node(
            "ber_spike",
            NodeType.ROOT_CAUSE,
            "Transient radio frequency interference or BER spike",
            degradation_modes=["emi_external", "solar_flare_interference"],
        )

        # Comms Intermediates

        self.add_node(
            "link_quality",
            NodeType.INTERMEDIATE,
            "Total RF link signal-to-noise ratio",
        )

        # Comms Observables

        self.add_node(
            "downlink_power_measured",
            NodeType.OBSERVABLE,
            "Measured downlink signal strength (dBm)",
        )

        self.add_node(
            "ber_measured",
            NodeType.OBSERVABLE,
            "Measured Bit Error Rate",
        )

        self.add_node(
            "transponder_temp_measured",
            NodeType.OBSERVABLE,
            "Measured transponder hardware temperature (C)",
        )

        # Comms Edges

        self.add_edge("transponder_fault", "link_quality", weight=0.85, mechanism="Reduced HPA gain lowers total SNR")
        self.add_edge("transponder_fault", "transponder_temp_measured", weight=0.7, mechanism="Inefficient HPA generates more waste heat")
        
        self.add_edge("antenna_pointing_error", "link_quality", weight=0.95, mechanism="Misalignment causes severe boresight signal loss")
        
        self.add_edge("ber_spike", "ber_measured", weight=0.98, mechanism="Direct observation of increased bit errors")
        
        self.add_edge("link_quality", "downlink_power_measured", weight=0.9, mechanism="Link SNR directly reflects in measured power")
        self.add_edge("link_quality", "ber_measured", weight=0.8, mechanism="Weak signal increases probability of bit errors")

    def _build_obc_subsystem_graph(self):
        """
        Build OBC (Onboard Computer) causal structure.
        
        WHY THIS MATTERS OPERATIONALLY:
        Differentiating between a 'busy' CPU and 'stuck' logic prevents
        unnecessary watchdog resets that could interrupt critical maneuvers.
        """

        # OBC Root Causes

        # ECSS-FM-OBC-01: Memory corruption
        self.add_node(
            "memory_corruption",
            NodeType.ROOT_CAUSE,
            "Single Event Upset or memory block corruption",
            degradation_modes=["seu", "multi_bit_fault"],
        )

        # ECSS-FM-OBC-02: Soft reset / Watchdog event
        self.add_node(
            "watchdog_reset_fault",
            NodeType.ROOT_CAUSE,
            "Unexplained watchdog timeout or system reset",
            degradation_modes=["loop_deadlock", "resource_starvation"],
        )

        # ECSS-FM-OBC-03: Software exception
        self.add_node(
            "software_exception",
            NodeType.ROOT_CAUSE,
            "Recurring software exceptions or task crashes",
            degradation_modes=["buffer_overflow", "logic_error"],
        )

        # OBC Intermediates

        self.add_node(
            "processor_state",
            NodeType.INTERMEDIATE,
            "Integrity of CPU execution and context",
        )

        # OBC Observables

        self.add_node(
            "cpu_load_measured",
            NodeType.OBSERVABLE,
            "Measured CPU usage percentage",
        )

        self.add_node(
            "memory_usage_measured",
            NodeType.OBSERVABLE,
            "Measured RAM usage percentage",
        )

        self.add_node(
            "reset_count_measured",
            NodeType.OBSERVABLE,
            "Cumulative OBC system reset count",
        )

        # OBC Edges

        self.add_edge("memory_corruption", "processor_state", weight=0.8, mechanism="Corrupt instructions or heap corrupts execution")
        self.add_edge("memory_corruption", "memory_usage_measured", weight=0.7, mechanism="Detection of leaked or locked memory blocks")
        
        self.add_edge("software_exception", "processor_state", weight=0.9, mechanism="Crashed tasks disrupt mission software")
        self.add_edge("software_exception", "cpu_load_measured", weight=0.6, mechanism="Error handlers and loggers consume cycles")

        self.add_edge("watchdog_reset_fault", "reset_count_measured", weight=1.0, mechanism="System logs every discrete reset event")
        
        self.add_edge("processor_state", "cpu_load_measured", weight=0.8, mechanism="Degraded software state often results in load spikes")
        self.add_edge("processor_state", "reset_count_measured", weight=0.5, mechanism="Corruption eventually triggers a reboot")

    def _build_propulsion_subsystem_graph(self):
        """
        Build Propulsion causal structure.
        
        WHY THIS MATTERS OPERATIONALLY:
        Propulsion is mission-critical for station-keeping. Distinguishing
        between a 'stuck' valve and a true 'leak' is the difference between
        a repairable software fix and a mission-ending catastrophe.
        """

        # Propulsion Root Causes

        # ECSS-FM-PROP-01: Thruster valve stuck
        self.add_node(
            "thruster_valve_fault",
            NodeType.ROOT_CAUSE,
            "Propellant valve stuck (open or closed)",
            degradation_modes=["mechanical_jam", "electric_coil_fault"],
        )

        # ECSS-FM-PROP-02: Fuel pressure leak
        self.add_node(
            "fuel_pressure_anomaly",
            NodeType.ROOT_CAUSE,
            "Anomaly in propellant tank or regulator pressure",
            degradation_modes=["seal_leak", "regulator_slip"],
        )

        # Intermediates: Propulsion

        self.add_node(
            "thrust_performance",
            NodeType.INTERMEDIATE,
            "Effective impulse vs commanded impulse",
        )

        # Observables: Propulsion

        self.add_node(
            "tank_pressure_measured",
            NodeType.OBSERVABLE,
            "Measured propellant tank pressure (PSI)",
        )

        self.add_node(
            "thruster_temp_measured",
            NodeType.OBSERVABLE,
            "Measured thruster nozzle temperature (C)",
        )

        # Edges: Propulsion

        self.add_edge("thruster_valve_fault", "thrust_performance", weight=0.95, mechanism="Stuck valve prevents or forces propellant flow")
        self.add_edge("thruster_valve_fault", "thruster_temp_measured", weight=0.8, mechanism="Valve state affects heat produced by combustion")
        
        self.add_edge("fuel_pressure_anomaly", "tank_pressure_measured", weight=0.98, mechanism="Leaking propellant directly reduces tank pressure")
        self.add_edge("fuel_pressure_anomaly", "thrust_performance", weight=0.7, mechanism="Variable pressure causes unstable thruster impulse")

    def _build_cross_subsystem_coupling(self):
        """Build edges representing cross-subsystem interactions."""

        # Power affects OBC stability
        self.add_edge("bus_regulation", "processor_state", weight=0.4, mechanism="Undervoltage causes CMOS latch-up or logic errors")
        
        # OBC affects ADCS (flight software runs loops)
        self.add_edge("processor_state", "pointing_accuracy", weight=0.5, mechanism="CPU overload increases control loop latency")

        # Propulsion affects Thermal (plume heating)
        self.add_edge("thrust_performance", "payload_temp", weight=0.3, mechanism="Plume impingement or conduction from engines heats payload")

    def add_node(
        self,
        name: str,
        node_type: NodeType,
        description: str,
        degradation_modes: List[str] = None,
    ):
        """
        Add a node to the graph.
        
        Args:
            name: Unique identifier for the node
            node_type: Whether this is a root cause, intermediate, or observable
            description: Human-readable explanation for operators
            degradation_modes: List of specific ways this node can fail
        """
        
        if degradation_modes is None:
            degradation_modes = []
        self.nodes[name] = Node(name, node_type, description, degradation_modes)

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        mechanism: str = "",
    ):
        """
        Add a directed causal edge to the graph.
        
        Args:
            source: Source node name (cause)
            target: Target node name (effect)
            weight: Strength of causal relationship (0-1, higher = stronger)
            mechanism: Explanation of how source causes target (for user interpretation)
            
        We validate that both nodes exist before adding the edge, preventing
        accidental dangling references that would cause inference to fail.
        """
        
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not in graph")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not in graph")

        self.edges.append(Edge(source, target, weight, mechanism))
        
        # Mirror in Rust core for fast traversal
        if self.rust_graph:
            self.rust_graph.add_edge(source, target, float(weight))

    def get_children(self, node_name: str) -> Dict[str, float]:
        """
        Get all children of a node (nodes it points to).
        
        This is used for forward inference: given a root cause, what effects propagate?
        
        Args:
            node_name: Node to query
            
        Returns:
            Dictionary mapping child node names to edge weights
        """
        
        children = {}
        for edge in self.edges:
            if edge.source == node_name:
                children[edge.target] = edge.weight
        return children

    def get_parents(self, node_name: str) -> Dict[str, float]:
        """
        Get all parents of a node (nodes pointing to it).
        
        This is used for backward inference: given an observable, what causes it?
        
        Args:
            node_name: Node to query
            
        Returns:
            Dictionary mapping parent node names to edge weights
        """
        
        parents = {}
        for edge in self.edges:
            if edge.target == node_name:
                parents[edge.source] = edge.weight
        return parents

    def get_root_causes(self) -> List[str]:
        """
        Get all root cause nodes.
        
        Returns:
            List of root cause node names (these are the diagnosis targets)
        """
        
        return [
            name
            for name, node in self.nodes.items()
            if node.node_type == NodeType.ROOT_CAUSE
        ]

    def get_observables(self) -> List[str]:
        """
        Get all observable (telemetry) nodes.
        
        Returns:
            List of observable node names (the measured quantities)
        """
        
        return [
            name
            for name, node in self.nodes.items()
            if node.node_type == NodeType.OBSERVABLE
        ]

    def get_weighted_paths_to_root(
        self, 
        node_name: str, 
        max_depth: int = 10
    ) -> List[Tuple[List[str], float]]:
        """
        Find all causal paths from a node back to root causes, including
        the cumulative causal strength (product of edge weights).
        
        Uses high-performance Rust core if available.
        """
        if self.rust_graph:
            return self.rust_graph.get_weighted_paths_to_root(node_name, max_depth)

        # Fallback to recursive Python implementation
        if max_depth == 0:
            return []

        parents = self.get_parents(node_name)
        if not parents:
            # We've reached a root cause
            return [([node_name], 1.0)]

        all_results = []
        for parent, weight in parents.items():
            parent_results = self.get_weighted_paths_to_root(parent, max_depth - 1)
            for path, parent_strength in parent_results:
                new_path = path + [node_name]
                all_results.append((new_path, parent_strength * weight))

        return all_results

    def get_paths_to_root(self, node_name: str, max_depth: int = 10) -> List[List[str]]:
        """
        Find all paths from a node back to root causes (upstream).
        This is a legacy method returning only paths (no weights).
        """
        weighted_results = self.get_weighted_paths_to_root(node_name, max_depth)
        return [path for path, strength in weighted_results]

    def print_structure(self):
        """Pretty-print graph structure for inspection."""
        
        print("\nCAUSAL GRAPH STRUCTURE")

        # Print nodes grouped by type
        for node_type in [NodeType.ROOT_CAUSE, NodeType.INTERMEDIATE, NodeType.OBSERVABLE]:
            nodes = [
                (name, node)
                for name, node in self.nodes.items()
                if node.node_type == node_type
            ]
            if nodes:
                print(f"\n{node_type.value.upper()}:")
                for name, node in sorted(nodes):
                    print(f"  • {name:25s} - {node.description}")
                    if node.degradation_modes:
                        modes_str = ", ".join(node.degradation_modes)
                        print(f"    Modes: {modes_str}")

        # Print all edges with weights and mechanisms
        print("\nCAUSAL EDGES:")
        for edge in sorted(self.edges, key=lambda e: e.source):
            print(
                f"  {edge.source:25s} → {edge.target:25s} "
                f"(weight={edge.weight:.2f})"
            )
            if edge.mechanism:
                print(f"    Mechanism: {edge.mechanism}")

        print("")


if __name__ == "__main__":
    # Quick test of graph structure
    graph = CausalGraph()
    graph.print_structure()

    # Example: Find all causal paths from a measurement back to root causes
    print("\nExample: Paths from battery_voltage_measured back to root causes:")
    paths = graph.get_paths_to_root("battery_voltage_measured")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {' ← '.join(reversed(path))}")
