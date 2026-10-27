"""Adam updates shared by the learning baselines."""
import numpy as np


class Adam:
    def __init__(self):
        self.steps = 0
        self.first = None
        self.second = None

    def update(self, weights, gradients, learning_rate):
        if self.first is None:
            self.first = [np.zeros_like(w) for w in weights]
            self.second = [np.zeros_like(w) for w in weights]
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        scale = min(1., 10. / max(norm, 1e-12))
        self.steps += 1
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            gradient = gradient * scale
            self.first[i] = .9 * self.first[i] + .1 * gradient
            self.second[i] = .999 * self.second[i] + .001 * gradient ** 2
            first = self.first[i] / (1 - .9 ** self.steps)
            second = self.second[i] / (1 - .999 ** self.steps)
            weight -= learning_rate * first / (np.sqrt(second) + 1e-8)
