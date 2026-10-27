"""Device topology, mobility, and server queue state."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from we_kb_fpa.config import NetworkConfig


@dataclass
class UserDevice:
    device_id: int
    x: float
    y: float
    z: float = 0.0
    p_tx: float = 0.2
    f_local: float = 1.0e9
    battery_remaining: float = 1000.0
    velocity: float = 0.0
    direction: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    pause_timer: float = 0.0


@dataclass
class EdgeServer:
    server_id: int
    x: float
    y: float
    z: float = 15.0
    f_total: float = 40.0e9
    bandwidth_total: float = 20.0e6
    cores: int = 8
    cpu_queue_load: float = 0.0
    queue_delay_current: float = 0.0


@dataclass
class CloudServer:
    server_id: int = 999
    f_total: float = 100.0e9
    propagation_delay: float = 0.050
    backhaul_bandwidth: float = 100.0e6
    cpu_queue_load: float = 0.0
    queue_delay_current: float = 0.0


class MobilityEngine:

    def __init__(self, config: NetworkConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng or np.random.default_rng(42)

    def step_rwp(self, user: UserDevice, dt: float):
        if user.pause_timer > 0.0:
            user.pause_timer = max(0.0, user.pause_timer - dt)
            return

        dx = user.target_x - user.x
        dy = user.target_y - user.y
        dist = np.hypot(dx, dy)

        step_dist = user.velocity * dt
        if dist <= step_dist or dist < 1e-4:

            user.x = user.target_x
            user.y = user.target_y
            user.pause_timer = self.rng.uniform(0.0, self.config.pause_time_max)
            user.target_x = self.rng.uniform(0.0, self.config.area_width)
            user.target_y = self.rng.uniform(0.0, self.config.area_height)
            user.velocity = self.rng.uniform(self.config.v_min, self.config.v_max)
            user.direction = np.arctan2(user.target_y - user.y, user.target_x - user.x)
        else:

            user.x += (dx / dist) * step_dist
            user.y += (dy / dist) * step_dist


        user.x = np.clip(user.x, 0.0, self.config.area_width)
        user.y = np.clip(user.y, 0.0, self.config.area_height)

    def step_gauss_markov(self, user: UserDevice, dt: float):
        alpha = self.config.gauss_markov_alpha
        v_mean = 0.5 * (self.config.v_min + self.config.v_max)
        d_mean = user.direction


        w_v = self.rng.normal(0.0, 1.0)
        w_d = self.rng.normal(0.0, 0.5)


        user.velocity = (
            alpha * user.velocity
            + (1.0 - alpha) * v_mean
            + np.sqrt(1.0 - alpha**2) * w_v
        )
        user.velocity = np.clip(user.velocity, self.config.v_min, self.config.v_max)

        user.direction = (
            alpha * user.direction
            + (1.0 - alpha) * d_mean
            + np.sqrt(1.0 - alpha**2) * w_d
        )

        user.x += user.velocity * np.cos(user.direction) * dt
        user.y += user.velocity * np.sin(user.direction) * dt


        if user.x < 0.0:
            user.x = -user.x
            user.direction = np.pi - user.direction
        elif user.x > self.config.area_width:
            user.x = 2 * self.config.area_width - user.x
            user.direction = np.pi - user.direction

        if user.y < 0.0:
            user.y = -user.y
            user.direction = -user.direction
        elif user.y > self.config.area_height:
            user.y = 2 * self.config.area_height - user.y
            user.direction = -user.direction

        user.x = np.clip(user.x, 0.0, self.config.area_width)
        user.y = np.clip(user.y, 0.0, self.config.area_height)


class NetworkEnvironment:

    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        seed: int = 42,
    ):
        self.config = config or NetworkConfig()
        self.rng = np.random.default_rng(seed)
        self.mobility_engine = MobilityEngine(self.config, self.rng)

        self.users: List[UserDevice] = []
        self.edge_servers: List[EdgeServer] = []
        self.cloud: CloudServer = CloudServer(
            f_total=self.config.f_cloud_total,
            propagation_delay=self.config.d_cloud_propagation,
            backhaul_bandwidth=self.config.backhaul_bandwidth,
        )
        self.current_time: float = 0.0
        self._initialize_topology()

    def _initialize_topology(self):

        M = self.config.num_edge_servers
        if M == 1:
            edge_positions = [(self.config.area_width / 2.0, self.config.area_height / 2.0)]
        elif M == 2:
            edge_positions = [
                (self.config.area_width * 0.25, self.config.area_height * 0.5),
                (self.config.area_width * 0.75, self.config.area_height * 0.5),
            ]
        elif M <= 4:
            xs = [self.config.area_width * 0.25, self.config.area_width * 0.75]
            ys = [self.config.area_height * 0.25, self.config.area_height * 0.75]
            edge_positions = [(x, y) for x in xs for y in ys][:M]
        else:

            grid_side = int(np.ceil(np.sqrt(M)))
            xs = np.linspace(50.0, self.config.area_width - 50.0, grid_side)
            ys = np.linspace(50.0, self.config.area_height - 50.0, grid_side)
            edge_positions = [(x, y) for x in xs for y in ys][:M]

        self.edge_servers = []
        for m_idx, (ex, ey) in enumerate(edge_positions):
            self.edge_servers.append(
                EdgeServer(
                    server_id=m_idx + 1,
                    x=ex,
                    y=ey,
                    z=15.0,
                    f_total=self.config.f_edge_total,
                    bandwidth_total=self.config.bandwidth_edge,
                    cores=self.config.edge_cores,
                )
            )


        self.users = []
        for i in range(self.config.num_users):
            ux = self.rng.uniform(0.0, self.config.area_width)
            uy = self.rng.uniform(0.0, self.config.area_height)
            ptx = self.rng.uniform(self.config.p_tx_min, self.config.p_tx_max)
            floc = self.rng.uniform(self.config.f_local_min, self.config.f_local_max)
            vel = self.rng.uniform(self.config.v_min, self.config.v_max)
            heading = self.rng.uniform(0.0, 2 * np.pi)

            user = UserDevice(
                device_id=i,
                x=ux,
                y=uy,
                z=1.5,
                p_tx=ptx,
                f_local=floc,
                battery_remaining=self.config.device_battery_capacity,
                velocity=vel,
                direction=heading,
                target_x=self.rng.uniform(0.0, self.config.area_width),
                target_y=self.rng.uniform(0.0, self.config.area_height),
                pause_timer=0.0,
            )
            self.users.append(user)

    def step_mobility(self, dt: float):
        self.current_time += dt
        for user in self.users:
            if self.config.mobility_model == "GaussMarkov":
                self.mobility_engine.step_gauss_markov(user, dt)
            else:
                self.mobility_engine.step_rwp(user, dt)

    def get_distance_matrix(self) -> np.ndarray:
        N = len(self.users)
        M = len(self.edge_servers)
        dist_mat = np.zeros((N, M), dtype=np.float64)

        for i, user in enumerate(self.users):
            for m, edge in enumerate(self.edge_servers):
                dx = user.x - edge.x
                dy = user.y - edge.y
                dz = user.z - edge.z
                # Avoid exact zero distance by setting a physical minimum antenna separation (1.0m)
                dist_mat[i, m] = max(1.0, np.sqrt(dx**2 + dy**2 + dz**2))

        return dist_mat

    def update_server_queues(self, dt: float):
        for edge in self.edge_servers:
            processed_cycles = edge.f_total * dt
            edge.cpu_queue_load = max(0.0, edge.cpu_queue_load - processed_cycles)
            edge.queue_delay_current = edge.cpu_queue_load / edge.f_total


        processed_cloud = self.cloud.f_total * dt
        self.cloud.cpu_queue_load = max(0.0, self.cloud.cpu_queue_load - processed_cloud)
        self.cloud.queue_delay_current = self.cloud.cpu_queue_load / self.cloud.f_total
