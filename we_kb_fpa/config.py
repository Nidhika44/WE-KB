"""Physical parameters and optimizer configurations; SI units unless stated."""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class NetworkConfig:

    area_width: float = 500.0
    area_height: float = 500.0


    num_users: int = 20          # N (IoT / User Devices)
    num_edge_servers: int = 4    # M (Edge Servers)
    num_cloud_servers: int = 1   # 1 Cloud Datacenter


    p_tx_min: float = 0.1        # Watts (20 dBm)
    p_tx_max: float = 0.5        # Watts (27 dBm)
    f_local_min: float = 0.8e9   # 0.8 GHz
    f_local_max: float = 1.5e9   # 1.5 GHz
    device_battery_capacity: float = 1000.0  # Joules


    f_edge_total: float = 40.0e9 # 40 GHz aggregate capacity per edge server
    bandwidth_edge: float = 20.0e6  # 20 MHz per edge server
    edge_cores: int = 8


    f_cloud_total: float = 100.0e9  # 100 GHz compute capacity
    d_cloud_propagation: float = 0.050  # 50 ms propagation delay
    backhaul_bandwidth: float = 100.0e6 # 100 Mbps fiber backhaul


    d_dl_constant: float = 0.002  # 2 ms downlink transfer delay


    kappa_local: float = 1.0e-28
    kappa_edge: float = 1.0e-30
    kappa_cloud: float = 1.0e-30


    mobility_model: str = "GaussMarkov"
    v_min: float = 1.0           # m/s (pedestrian)
    v_max: float = 15.0          # m/s (vehicular)
    pause_time_max: float = 2.0  # seconds
    gauss_markov_alpha: float = 0.75  # memory factor in [0, 1]


@dataclass
class ChannelConfig:

    carrier_frequency: float = 3.5e9    # 3.5 GHz (5G-NR sub-6 GHz band)
    speed_of_light: float = 3.0e8       # m/s
    noise_power_spectral_density_dbm: float = -174.0 # dBm/Hz
    noise_figure_db: float = 5.0        # Receiver noise figure (dB)


    reference_distance: float = 1.0     # d_0 = 1 meter
    path_loss_exponent: float = 3.5     # Urban microcellular environment
    pl_0_db: float = 38.4               # Path loss at d_0 = 1m for 3.5 GHz
    shadowing_std_db: float = 6.0       # Log-normal shadowing std dev


    fading_model: str = "Rayleigh"      # 'Rayleigh' or 'Rician'
    rician_k_factor_db: float = 6.0     # K-factor for Rician fading (LOS)
    coherence_time: float = 0.010       # 10 ms channel coherence interval


@dataclass
class TaskConfig:

    d_k_min_mb: float = 0.5             # 0.5 Mbits
    d_k_max_mb: float = 2.5             # 2.5 Mbits


    c_k_min: float = 300.0              # 300 cycles/bit
    c_k_max: float = 700.0              # 700 cycles/bit


    tau_max_min: float = 0.300          # 300 ms
    tau_max_max: float = 0.700          # 700 ms


    lambda_base: float = 5.0            # Base arrival rate (tasks/sec)
    lambda_peak_multiplier: float = 2.5 # Peak arrival surge factor
    burst_period: float = 20.0          # Time period for diurnal/burst cycle (sec)


@dataclass
class CostConfig:

    alpha: float = 0.5


    penalty_deadline_multiplier: float = 50.0    # Penalty for QoS deadline breach
    penalty_cpu_multiplier: float = 50.0         # Penalty for edge CPU capacity overflow
    penalty_bandwidth_multiplier: float = 50.0   # Penalty for bandwidth allocation > 100%


@dataclass
class KnowledgeBaseConfig:
    memory_capacity: int = 5000         # Max historical state-solution pairs
    k_neighbors: int = 5                # K nearest neighbors in KNN
    similarity_metric: str = "minkowski"# Metric ('euclidean', 'cosine', 'minkowski')
    min_confidence_score: float = 0.70  # Pruning confidence threshold
    feature_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])


@dataclass
class OptimizerConfig:
    population_size: int = 40           # Swarm/flower population
    max_iterations: int = 80            # Max optimization generations per slot


    base_switch_prob: float = 0.8       # p_base (global vs local pollination)
    levy_exponent: float = 1.5          # lambda for Levy flight
    gamma_base: float = 0.05            # Base Levy step size


    csi_variance_theta: float = 0.5     # Normalizing constant theta for dynamic p_s
    sinr_ref_db: float = 10.0           # Reference SINR for step size scaling
    sinr_scaling_beta: float = 1.5      # Adaptive jump magnitude scaling factor
    kb_seeding_percentage: float = 0.30 # Percentage of population seeded from KB (30%)


    pso_w: float = 0.7                  # Inertia weight
    pso_c1: float = 1.5                 # Cognitive acceleration
    pso_c2: float = 1.5                 # Social acceleration


    ga_crossover_prob: float = 0.85
    ga_mutation_prob: float = 0.10
    ga_tournament_size: int = 3


@dataclass
class SimulationConfig:
    time_step: float = 0.1              # Discrete simulation step (100 ms)
    total_slots: int = 50               # Number of simulation slots per run
    monte_carlo_runs: int = 100         # Runs for statistical significance
    random_seed: int = 42               # Master seed
