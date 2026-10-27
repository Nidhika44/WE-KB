"""Task distributions and optional NHPP arrival generation."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from we_kb_fpa.config import TaskConfig


@dataclass
class Task:
    task_id: int
    user_id: int
    data_size_bits: float
    cycles_per_bit: float
    deadline_sec: float
    arrival_time: float = 0.0

    @property
    def total_cycles(self) -> float:
        return self.data_size_bits * self.cycles_per_bit

    @property
    def data_size_mb(self) -> float:
        return self.data_size_bits / 1.0e6


class TaskWorkloadGenerator:

    def __init__(self, config: Optional[TaskConfig] = None, seed: int = 42):
        self.config = config or TaskConfig()
        self.rng = np.random.default_rng(seed)
        self._task_counter: int = 0

    def get_arrival_rate(self, current_time: float) -> float:
        period = self.config.burst_period
        peak_mult = self.config.lambda_peak_multiplier
        oscillation = np.sin(np.pi * current_time / period) ** 2
        return self.config.lambda_base * (1.0 + (peak_mult - 1.0) * oscillation)

    def generate_slot_tasks(self, num_users: int, current_time: float) -> List[Task]:
        """Generates a task for each active user in the current decision slot.

        Attribute distributions:
        - D_k ~ Uniform(D_min, D_max) in bits
        - C_k ~ Uniform(C_min, C_max) in cycles/bit
        - tau_max ~ Uniform(tau_min, tau_max) in seconds
        """
        tasks = []
        for u_id in range(num_users):
            self._task_counter += 1

            d_mb = self.rng.uniform(self.config.d_k_min_mb, self.config.d_k_max_mb)
            d_bits = d_mb * 1.0e6


            c_val = self.rng.uniform(self.config.c_k_min, self.config.c_k_max)


            tau_max = self.rng.uniform(self.config.tau_max_min, self.config.tau_max_max)

            tasks.append(
                Task(
                    task_id=self._task_counter,
                    user_id=u_id,
                    data_size_bits=d_bits,
                    cycles_per_bit=c_val,
                    deadline_sec=tau_max,
                    arrival_time=current_time,
                )
            )
        return tasks

    def generate_nhpp_arrivals(
        self, num_users: int, start_time: float, duration: float
    ) -> List[Task]:
        max_rate = self.config.lambda_base * self.config.lambda_peak_multiplier
        t = start_time
        arrival_tasks = []

        while t < start_time + duration:

            delta_t = self.rng.exponential(1.0 / max_rate)
            t += delta_t
            if t > start_time + duration:
                break


            instant_rate = self.get_arrival_rate(t)
            if self.rng.uniform(0.0, max_rate) <= instant_rate:

                u_id = int(self.rng.integers(0, num_users))
                self._task_counter += 1
                d_mb = self.rng.uniform(self.config.d_k_min_mb, self.config.d_k_max_mb)
                c_val = self.rng.uniform(self.config.c_k_min, self.config.c_k_max)
                tau_max = self.rng.uniform(self.config.tau_max_min, self.config.tau_max_max)

                arrival_tasks.append(
                    Task(
                        task_id=self._task_counter,
                        user_id=u_id,
                        data_size_bits=d_mb * 1.0e6,
                        cycles_per_bit=c_val,
                        deadline_sec=tau_max,
                        arrival_time=t,
                    )
                )

        return arrival_tasks
