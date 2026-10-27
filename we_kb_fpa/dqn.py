"""NumPy DQN: uniform replay, frozen target network, epsilon-greedy, Huber loss.

Mnih et al. (2015), https://doi.org/10.1038/nature14236.
Each slot is a finite episode. Dense potential differences telescope to the
negative scaled terminal cost when gamma=1. The legacy log transform is an
explicit option and does not optimize expected raw cost. Observations include the full slot
and partial assignment, so resource interactions are visible to the policy.
"""
from collections import deque
from dataclasses import dataclass
import time
import numpy as np
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer, OptimizationOutput
from we_kb_fpa.allocation import uniform_allocation, slot_features, task_features, action_features
from we_kb_fpa.neural import Adam


@dataclass
class DQNConfig:
    hidden: int = 64
    batch_size: int = 32
    replay_capacity: int = 10000
    target_interval: int = 100
    learning_rate: float = 0.001
    gamma: float = 1.0
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    reward_transform: str = "linear"
    reward_scale: float = 100.


class DQNBaseline:
    def __init__(self, opt_config=None, cost_config=None, net_config=None,
                 seed=42, config=None):
        self.config = config or DQNConfig()
        c = self.config
        if (c.hidden < 1 or c.batch_size < 1 or c.replay_capacity < c.batch_size
                or c.target_interval < 1 or c.learning_rate <= 0
                or not 0 <= c.gamma <= 1 or not 0 <= c.epsilon_min <= c.epsilon <= 1
                or not 0 < c.epsilon_decay <= 1 or c.reward_transform not in ('linear','log')
                or not np.isfinite(c.reward_scale) or c.reward_scale<=0):
            raise ValueError("Invalid DQN configuration")
        self.rng = np.random.default_rng(seed)
        self.decoder = WEKBFPAOptimizer(opt_config, cost_config, net_config, use_kb=False)
        self.memory = deque(maxlen=c.replay_capacity)
        self.weights = None
        self.epsilon = c.epsilon
        self.updates = 0
        self.losses = []
        self.adam = Adam()

    def _initialize(self, inputs, actions):
        h = self.config.hidden
        self.weights = [self.rng.normal(0, np.sqrt(2 / inputs), (inputs, h)),
                        np.zeros(h), self.rng.normal(0, np.sqrt(2 / h), (h, h)),
                        np.zeros(h), self.rng.normal(0, 0.01, (h, actions)), np.zeros(actions)]
        self.target = [w.copy() for w in self.weights]

    @staticmethod
    def _forward(x, w):
        h1 = np.maximum(0, x @ w[0] + w[1])
        h2 = np.maximum(0, h1 @ w[2] + w[3])
        return h2 @ w[4] + w[5], h1, h2

    def learn(self):
        c = self.config
        if len(self.memory) < c.batch_size:
            return
        batch = [self.memory[i] for i in self.rng.choice(len(self.memory), c.batch_size, replace=False)]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*batch))
        w = self.weights
        q, h1, h2 = self._forward(states, w)
        next_q = self._forward(next_states, self.target)[0].max(axis=1)
        expected = rewards + c.gamma * (1 - done) * next_q
        err = q[np.arange(c.batch_size), actions] - expected
        self.losses.append(float(np.mean(np.where(abs(err) <= 1, .5 * err**2, abs(err) - .5))))
        dq = np.zeros_like(q)
        dq[np.arange(c.batch_size), actions] = np.clip(err, -1, 1) / c.batch_size
        dh2 = (dq @ w[4].T) * (h2 > 0)
        dh1 = (dh2 @ w[2].T) * (h1 > 0)
        grads = [states.T @ dh1, dh1.sum(0), h1.T @ dh2, dh2.sum(0), h2.T @ dq, dq.sum(0)]
        self.adam.update(w, grads, c.learning_rate)
        self.updates += 1
        if self.updates % c.target_interval == 0:
            self.target = [weight.copy() for weight in w]

    def run(self, tasks, users, edges, cloud, sinr, distances, training=False, explore=True):
        start = time.perf_counter()
        n, m = len(tasks), len(edges)
        if n == 0 or m == 0:
            raise ValueError("DQN requires tasks and at least one edge")
        actions_count = m + 2
        targets = np.zeros(n, dtype=int)
        assigned = np.zeros((n, actions_count))
        context = slot_features(tasks, users, edges, cloud, sinr, distances)
        current = task_features(tasks, users, sinr, distances)
        radio_counts = np.zeros(m)
        home = np.argmin(distances, axis=1)

        def state(i):
            counts = assigned.sum(axis=0)
            estimates = action_features(tasks[i], users[i], edges, cloud, sinr[i], distances[i],
                                        counts, radio_counts, self.decoder.net_cfg)
            return np.concatenate(([i / n], current[i], estimates, counts/n, context, assigned.ravel()))

        def evaluate():
            allocation = uniform_allocation(targets, tasks, users, edges, cloud, distances)
            result = self.decoder.cost_engine.evaluate_allocation(
                tasks, users, edges, cloud, *allocation, sinr, distances)
            return allocation, result

        initial = state(0)
        if self.weights is None:
            self._initialize(len(initial), actions_count)
        if self.weights[0].shape[0] != len(initial) or self.weights[-1].size != actions_count:
            raise ValueError("DQN topology dimensions changed; create a new agent")
        transitions = []
        def potential(cost):
            return cost/(n*self.config.reward_scale) if self.config.reward_transform=='linear' else np.log1p(cost/n)
        previous = potential(evaluate()[1].total_cost) if training else None
        for i in range(n):
            s = state(i)
            action = (int(self.rng.integers(actions_count)) if training and explore and self.rng.random() < self.epsilon
                      else int(np.argmax(self._forward(s, self.weights)[0])))
            targets[i] = action
            assigned[i, action] = 1.
            if action:
                radio_counts[action-1 if action <= m else home[i]] += 1
            if training:
                current_potential = potential(evaluate()[1].total_cost)
                reward = float(previous - current_potential)
                previous = current_potential
                transitions.append((s, action, reward,
                                    state(i + 1) if i + 1 < n else np.zeros_like(s), i == n - 1))
        (tgt, cpu, bw), result = evaluate()
        inference_seconds = time.perf_counter() - start
        if training:
            for transition in transitions:
                self.memory.append(transition)
                self.learn()
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        self.last_decision_seconds = inference_seconds
        self.last_learning_seconds = time.perf_counter() - start - inference_seconds if training else 0.
        return OptimizationOutput(result.total_cost, tgt, cpu, bw, [result.total_cost],
                                  time.perf_counter() - start, 1, result, "DQN")
