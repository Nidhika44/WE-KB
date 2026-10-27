"""Shared stopping heuristic for the fixed-snapshot decision-delay model.

Waiting cost is exact for a fixed incumbent. Predicted search improvement
uses recent observations and an optimism bonus, not a statistical guarantee.
"""
from collections import deque
import numpy as np


class DelayProfile:
    def __init__(self, output, cost_config):
        self.cost = output.best_cost
        self.alpha = cost_config.alpha
        self.penalty = cost_config.penalty_deadline_multiplier
        metrics = output.evaluation_result.task_metrics
        self.inverse_deadlines = 1./np.array([m.deadline_sec for m in metrics])
        self.ratios = np.array([m.latency_total for m in metrics])*self.inverse_deadlines
        self.base_penalties = np.square(self.ratios[self.ratios>1.]).sum()
        self.latency_slope = self.alpha*self.inverse_deadlines.sum()

    def score(self, delay):
        ratios = self.ratios+delay*self.inverse_deadlines
        return float(self.cost+delay*self.latency_slope+self.penalty*(np.square(ratios[ratios>1.]).sum()-self.base_penalties))

    def waiting_cost(self, delay, additional):
        return max(0.,self.score(delay+additional)-self.score(delay))


class ComputationController:
    def __init__(self, window=16, minimum=8, optimism=1., forecast_factor=1.5):
        if window<2 or not 1<=minimum<=window or not np.isfinite(optimism) or optimism<0 or not np.isfinite(forecast_factor) or forecast_factor<1:
            raise ValueError('Invalid computation controller')
        self.gains = deque(maxlen=window)
        self.durations = deque(maxlen=window)
        self.minimum,self.optimism,self.factor = minimum,optimism,forecast_factor
        self.last = {}

    def observe(self, improvement, seconds):
        if not np.isfinite(improvement) or improvement<0 or not np.isfinite(seconds) or seconds<0:
            raise ValueError('Search observations must be finite and nonnegative')
        self.gains.append(improvement)
        self.durations.append(max(seconds,1e-9))

    def continue_search(self, profile, elapsed):
        count = len(self.gains)
        if count<self.minimum:
            return True
        gains = np.asarray(self.gains)
        forecast = self.factor*float(np.mean(self.durations))
        benefit = float(gains.mean()+self.optimism*gains.std(ddof=1)/np.sqrt(count)) if count>1 else float(gains.mean())
        waiting = profile.waiting_cost(elapsed,forecast)
        self.last = dict(predicted_improvement=benefit,predicted_waiting_cost=waiting,
                         predicted_step_ms=forecast*1000,stopping_samples=count)
        return benefit>waiting
