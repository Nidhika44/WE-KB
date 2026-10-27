"""DROO-Adapted: order-preserving binary candidates, replay, cross-entropy, Adam.

Reference: Huang, Bi, Zhang, https://arxiv.org/abs/1808.01977, Sec. IV-B/C.
Adaptations: each user has a nearest-edge home association; binary local/edge
decisions minimize this simulator's penalized cost with shared uniform shares.
Inputs include workload/device state. Wireless power transfer and the paper's
convex time-allocation subproblem are absent. K=N+1 is fixed, not adaptive.
This is a literature-based adaptation, not reproduction of published results.
"""
from collections import deque
import time
import numpy as np
from we_kb_fpa.allocation import slot_features, uniform_allocation
from we_kb_fpa.dqn import DQNBaseline
from we_kb_fpa.neural import Adam
from we_kb_fpa.we_kb_fpa import OptimizationOutput
from we_kb_fpa.cost_engine import CostEngine


def order_preserving_candidates(probabilities):
    p = np.asarray(probabilities)
    result = [(p > .5).astype(int)]
    for index in np.argsort(abs(p - .5), kind="stable"):
        threshold = p[index]
        result.append(((p > threshold) | ((p == threshold) & (threshold <= .5))).astype(int))
    return np.unique(result, axis=0)


class DROOAdapted:
    def __init__(self, cost_config=None, net_config=None, seed=42, hidden=64, batch_size=32):
        self.rng = np.random.default_rng(seed)
        self.engine = CostEngine(cost_config, net_config)
        self.hidden, self.batch_size = hidden, batch_size
        self.memory = deque(maxlen=1024)
        self.weights = None
        self.adam = Adam()
        self.updates = 0
        self.losses = []

    @staticmethod
    def probabilities(x, weights):
        logits, h1, h2 = DQNBaseline._forward(x, weights)
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40))), h1, h2

    def run(self, tasks, users, edges, cloud, sinr, distances, training=False):
        start = time.perf_counter()
        n = len(tasks)
        x = slot_features(tasks, users, edges, cloud, sinr, distances)
        if self.weights is None:
            h = self.hidden
            self.weights = [self.rng.normal(0, np.sqrt(2 / len(x)), (len(x), h)), np.zeros(h),
                            self.rng.normal(0, np.sqrt(2 / h), (h, h)), np.zeros(h),
                            self.rng.normal(0, .01, (h, n)), np.zeros(n)]
        if self.weights[-1].size != n or self.weights[0].shape[0] != len(x):
            raise ValueError("DROO-Adapted requires fixed observation dimensions")
        probabilities = self.probabilities(x, self.weights)[0]
        home = np.argmin(distances, axis=1) + 1
        best = None
        for bits in order_preserving_candidates(probabilities):
            allocation = uniform_allocation(bits * home, tasks, users, edges, cloud, distances)
            evaluation = self.engine.evaluate_allocation(tasks, users, edges, cloud, *allocation, sinr, distances)
            if best is None or evaluation.total_cost < best[0].total_cost:
                best = evaluation, allocation, bits
        self.last_decision_seconds = time.perf_counter() - start
        if training:
            self.memory.append((x.copy(), best[2].copy()))
            if len(self.memory) >= self.batch_size:
                batch = [self.memory[i] for i in self.rng.choice(len(self.memory), self.batch_size, replace=False)]
                xs, labels = map(np.asarray, zip(*batch))
                p, h1, h2 = self.probabilities(xs, self.weights)
                p = np.clip(p, 1e-8, 1 - 1e-8)
                self.losses.append(float(-np.mean(labels * np.log(p) + (1-labels) * np.log1p(-p))))
                dz = (p - labels) / (self.batch_size * n)
                dh2 = (dz @ self.weights[4].T) * (h2 > 0)
                dh1 = (dh2 @ self.weights[2].T) * (h1 > 0)
                gradients = [xs.T @ dh1, dh1.sum(0), h1.T @ dh2, dh2.sum(0), h2.T @ dz, dz.sum(0)]
                self.adam.update(self.weights, gradients, .001)
                self.updates += 1
        duration = time.perf_counter() - start
        self.last_learning_seconds = duration - self.last_decision_seconds if training else 0.
        evaluation, (targets, cpu, bw), _ = best
        return OptimizationOutput(evaluation.total_cost, targets, cpu, bw, [evaluation.total_cost],
                                  duration, 1, evaluation, "DROO-Adapted")
