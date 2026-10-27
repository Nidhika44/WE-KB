"""Path loss, shadowing, fading, and multi-user SINR."""

from typing import Optional, Tuple
import numpy as np
from we_kb_fpa.config import ChannelConfig, NetworkConfig


class WirelessChannelSimulator:

    def __init__(
        self,
        channel_config: Optional[ChannelConfig] = None,
        net_config: Optional[NetworkConfig] = None,
        seed: int = 42,
    ):
        self.ch_cfg = channel_config or ChannelConfig()
        self.net_cfg = net_config or NetworkConfig()
        self.rng = np.random.default_rng(seed)


        bw_hz = self.net_cfg.bandwidth_edge
        noise_dbm = (
            self.ch_cfg.noise_power_spectral_density_dbm
            + 10.0 * np.log10(bw_hz)
            + self.ch_cfg.noise_figure_db
        )
        self.noise_power_watts = 10.0 ** ((noise_dbm - 30.0) / 10.0)


        self.last_h_coefficients: Optional[np.ndarray] = None
        self.last_path_loss_linear: Optional[np.ndarray] = None

    def compute_path_loss_db(self, distances: np.ndarray) -> np.ndarray:

        d = np.maximum(distances, self.ch_cfg.reference_distance)
        pl_deterministic = self.ch_cfg.pl_0_db + 10.0 * self.ch_cfg.path_loss_exponent * np.log10(d)


        shadowing = self.rng.normal(0.0, self.ch_cfg.shadowing_std_db, size=distances.shape)
        pl_db = pl_deterministic + shadowing
        return pl_db

    def sample_small_scale_fading(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.ch_cfg.fading_model == "Rayleigh":

            h_real = self.rng.normal(0.0, np.sqrt(0.5), size=shape)
            h_imag = self.rng.normal(0.0, np.sqrt(0.5), size=shape)
            fading_power = h_real**2 + h_imag**2
        else:

            k_lin = 10.0 ** (self.ch_cfg.rician_k_factor_db / 10.0)
            mu_los = np.sqrt(k_lin / (k_lin + 1.0))
            sigma_nlos = np.sqrt(1.0 / (2.0 * (k_lin + 1.0)))
            h_real = self.rng.normal(mu_los, sigma_nlos, size=shape)
            h_imag = self.rng.normal(0.0, sigma_nlos, size=shape)
            fading_power = h_real**2 + h_imag**2

        self.last_h_coefficients = fading_power
        return fading_power

    def update_channel_states(self, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pl_db = self.compute_path_loss_db(distances)

        pl_linear = 10.0 ** (pl_db / 10.0)
        self.last_path_loss_linear = pl_linear

        fading_gains = self.sample_small_scale_fading(distances.shape)

        composite_gains = fading_gains / np.maximum(pl_linear, 1e-12)
        return composite_gains, fading_gains

    def compute_sinr_matrix(
        self,
        composite_gains: np.ndarray,
        p_tx_vector: np.ndarray,
        offloading_targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        N, M = composite_gains.shape
        sinr_matrix = np.zeros((N, M), dtype=np.float64)


        rx_power = p_tx_vector[:, np.newaxis] * composite_gains

        for m in range(M):

            if offloading_targets is not None:

                active_users_to_m = np.where(offloading_targets == (m + 1))[0]
            else:
                active_users_to_m = np.arange(N)

            for i in range(N):
                sig = rx_power[i, m]

                if offloading_targets is not None:
                    interf_users = [u for u in active_users_to_m if u != i]
                    interference = np.sum(rx_power[interf_users, m]) if len(interf_users) > 0 else 0.0
                else:

                    interference = 0.05 * np.sum(rx_power[:, m]) - 0.05 * sig
                    interference = max(0.0, interference)

                sinr_linear = sig / (self.noise_power_watts + interference)
                sinr_matrix[i, m] = max(1e-6, sinr_linear)

        return sinr_matrix

    def compute_data_rates(
        self,
        sinr_matrix: np.ndarray,
        bandwidth_allocations: np.ndarray,
    ) -> np.ndarray:
        B_m = self.net_cfg.bandwidth_edge

        rates = bandwidth_allocations * B_m * np.log2(1.0 + sinr_matrix)

        return np.maximum(1e3, rates)

    def get_channel_variance(self) -> float:
        if self.last_h_coefficients is None:
            return 0.5
        return float(np.var(self.last_h_coefficients))
