"""Budgeted categorical search with greedy elitism and snapshot warm starts.

Uniform shares are feasible allocations, not claimed analytical optima.
Wall-clock limits are soft: an in-flight evaluation cannot be interrupted.
Deadline-aware deployment ranks up to eight observed incumbents by delayed
cost, restricted to raw cost no worse than the same-snapshot greedy seed.
Its final raw-cost trace sample can exceed the search's raw minimum.
"""
from dataclasses import dataclass
import time
import numpy as np
from we_kb_fpa.allocation import uniform_allocation, binary_greedy_allocation
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.config import OptimizerConfig
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer, OptimizationOutput
from we_kb_fpa.computation_control import ComputationController,DelayProfile


@dataclass(frozen=True)
class AnytimeConfig:
    budget_ms: float = 10.
    population: int = 12
    patience: int = 40
    max_evaluations: int = 10000
    use_kb: bool = True
    warm_start: bool = True
    event_trigger: bool = True
    channel_threshold_db: float = 3.
    workload_threshold: float = .15
    strategy: str = "fpa"
    action_scope: str = "all"
    deadline_aware: bool = False
    stopping_enabled: bool = True
    stop_window: int = 16
    stop_min_samples: int = 8
    stop_optimism: float = 1.
    forecast_factor: float = 1.5

    def __post_init__(self):
        if (not np.isfinite(self.budget_ms) or self.budget_ms <= 0 or self.population < 2
                or self.patience < 1 or self.max_evaluations < 1
                or not np.isfinite(self.channel_threshold_db) or self.channel_threshold_db <= 0
                or not np.isfinite(self.workload_threshold) or self.workload_threshold <= 0
                or self.strategy not in ("fpa", "pso", "ga", "anchor")
                or self.action_scope not in ("all", "binary_nearest")):
            raise ValueError("Invalid anytime optimizer settings")
        ComputationController(self.stop_window,self.stop_min_samples,self.stop_optimism,self.forecast_factor)


class AnytimeWEKBFPA:
    def __init__(self, config=None, net_config=None, cost_config=None, seed=42):
        self.config = config or AnytimeConfig()
        self.runner = BaselineRunner(net_config=net_config, cost_config=cost_config, seed=seed)
        self.guide = WEKBFPAOptimizer(OptimizerConfig(), cost_config, net_config, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.previous_targets = None
        self.previous_context = None
        self.last_diagnostics = {}

    def optimize(self, tasks, users, edges, cloud, sinr, distances,
                 channel_variance=.5, knowledge_base=None):
        started = time.perf_counter()
        cfg, n, choices = self.config, len(tasks), len(edges) + 2
        if not n:
            raise ValueError("At least one task is required")
        sinr_db = 10 * np.log10(np.maximum(sinr, 1e-12))
        workload = np.array([t.total_cycles / t.deadline_sec for t in tasks])
        context = (sinr_db.copy(), workload)
        trigger = self.previous_context is None
        if not trigger:
            old_channel, old_workload = self.previous_context
            trigger = (old_channel.shape != sinr_db.shape or old_workload.shape != workload.shape)
            if not trigger:
                trigger = (np.median(np.abs(sinr_db-old_channel)) >= cfg.channel_threshold_db
                           or np.mean(np.abs(workload-old_workload)/np.maximum(old_workload, 1.)) >= cfg.workload_threshold)

        binary = cfg.action_scope == "binary_nearest"
        home = np.argmin(distances,axis=1)+1
        def project(targets):
            targets = np.asarray(targets,dtype=np.int32)
            return np.where(targets==0,0,home).astype(np.int32) if binary else targets.copy()
        if binary:
            choices = 2
            allocation = binary_greedy_allocation(tasks,users,edges,cloud,sinr,distances,self.runner.net_cfg)
            result = self.runner.cost_engine.evaluate_allocation(tasks,users,edges,cloud,*allocation,sinr,distances)
            best = OptimizationOutput(result.total_cost,*allocation,[],0.,0,result,"Binary greedy seed")
        else:
            best = self.runner.run_greedy_latency(tasks, users, edges, cloud, sinr, distances)
        greedy_cost = best.best_cost
        cache = {best.best_targets.astype(np.int32).tobytes(): best.best_cost}
        population, scores = [best.best_targets.copy()], [best.best_cost]
        evaluations, stalled, longest = 1, 0, .0003
        curve = [best.best_cost]
        proposals, cache_hits, improvements, changed_users = 0, 0, 0, 0
        controller = None
        search_clock = None
        archive = [(best,DelayProfile(best,self.runner.cost_cfg))] if cfg.deadline_aware else []
        stop_reason = 'budget_or_evaluation_limit'
        value_stopped = False

        def deployment(elapsed):
            return min(archive,key=lambda item:item[1].score(elapsed))

        def remaining():
            nonlocal stop_reason,value_stopped
            if value_stopped:
                return False
            elapsed = time.perf_counter()-started
            if evaluations>=cfg.max_evaluations or elapsed+2*longest>=cfg.budget_ms/1000:
                stop_reason = 'budget_or_evaluation_limit'
                return False
            if controller is not None and not controller.continue_search(deployment(elapsed)[1],elapsed):
                stop_reason = 'estimated_gain_below_waiting_cost'
                value_stopped = True
                return False
            return True

        def evaluate(candidate):
            nonlocal best, evaluations, longest, stalled, cache_hits, improvements
            key = candidate.astype(np.int32).tobytes()
            if key in cache:
                cache_hits += 1
                return cache[key]
            tick = time.perf_counter()
            allocation = uniform_allocation(candidate, tasks, users, edges, cloud, distances)
            result = self.runner.cost_engine.evaluate_allocation(
                tasks, users, edges, cloud, *allocation, sinr, distances)
            if cfg.deadline_aware and result.total_cost<=greedy_cost:
                candidate_output = OptimizationOutput(result.total_cost,*allocation,[],0.,0,result,'Deadline-aware incumbent')
                archive.append((candidate_output,DelayProfile(candidate_output,self.runner.cost_cfg)))
                if len(archive)>8:
                    now = time.perf_counter()-started
                    archive.sort(key=lambda item:item[1].score(now))
                    del archive[8:]
            evaluations += 1
            stalled += 1
            if result.total_cost < best.best_cost:
                best = OptimizationOutput(result.total_cost, *allocation, [], 0., 0, result, "Anytime-WE-KB-FPA")
                stalled = 0
                improvements += 1
            cache[key] = result.total_cost
            curve.append(best.best_cost)
            longest = max(longest, time.perf_counter()-tick)
            return result.total_cost

        anchors = []
        if cfg.warm_start and self.previous_targets is not None and len(self.previous_targets) == n:
            anchors.append(self.previous_targets.copy())
        anchors.append(np.zeros(n, dtype=np.int32))
        if cfg.use_kb and knowledge_base is not None and knowledge_base.records and remaining():
            anchors.extend(rec.solution_targets.copy() for rec, _ in
                           knowledge_base.query_nearest_solutions(sinr, tasks, distances, k=3)
                           if len(rec.solution_targets) == n)
        for candidate in anchors:
            if remaining() and np.all((candidate >= 0) & (candidate < len(edges)+2)):
                candidate = project(candidate)
                scores.append(evaluate(candidate))
                population.append(candidate)
        search = cfg.strategy != "anchor" and (not cfg.event_trigger or trigger)
        anchor_cost = best.best_cost
        anchor_improvements = improvements
        if cfg.deadline_aware and cfg.stopping_enabled and search:
            controller = ComputationController(cfg.stop_window,cfg.stop_min_samples,cfg.stop_optimism,cfg.forecast_factor)
            search_clock = time.perf_counter()

        def record_search(previous_profiles):
            nonlocal search_clock
            if controller is not None:
                now = time.perf_counter()-started
                improvement = max(0.,min(profile.score(now) for profile in previous_profiles)-deployment(now)[1].score(now))
                controller.observe(improvement,time.perf_counter()-search_clock)
                search_clock = time.perf_counter()

        while search and len(population) < cfg.population and remaining():
            previous_profiles = [item[1] for item in archive]
            candidate = best.best_targets.copy()
            mutate = self.rng.random(n) < .3
            random_targets = self.rng.integers(choices,size=n)
            if binary:
                random_targets = np.where(random_targets==0,0,home)
            candidate[mutate] = random_targets[mutate]
            population.append(candidate.astype(np.int32))
            scores.append(evaluate(candidate))
            record_search(previous_profiles)
        personal = [p.copy() for p in population]
        personal_scores = list(scores)
        velocities = np.zeros((len(population), n, choices)) if cfg.strategy == "pso" else None
        switch = self.guide._compute_dynamic_switch_prob(channel_variance)
        attempts = 0
        while search and remaining() and stalled < cfg.patience and attempts < 10*cfg.max_evaluations:
            previous_profiles = [item[1] for item in archive]
            attempts += 1
            i = int(self.rng.integers(len(population)))
            candidate = population[i].copy()
            if cfg.strategy == "pso":
                encode = lambda targets: (targets!=0).astype(int) if binary else targets
                onehot = np.eye(choices)[encode(candidate)]
                velocities[i] = (.7*velocities[i] + 1.5*self.rng.random((n, 1))*(np.eye(choices)[encode(personal[i])]-onehot)
                                 + 1.5*self.rng.random((n, 1))*(np.eye(choices)[encode(best.best_targets)]-onehot))
                logits = velocities[i] - velocities[i].max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                candidate = np.array([self.rng.choice(choices, p=p) for p in probabilities], dtype=np.int32)
                if binary:
                    candidate = np.where(candidate==0,0,home).astype(np.int32)
            elif cfg.strategy == "ga":
                parents = []
                for _ in range(2):
                    tournament = self.rng.integers(len(population), size=3)
                    parents.append(population[min(tournament, key=lambda j: scores[j])])
                candidate = np.where(self.rng.random(n) < .5, parents[0], parents[1]).astype(np.int32)
            elif self.rng.random() < switch:
                scales = self.guide._compute_channel_guided_step_sizes(sinr, candidate)
                probability = np.clip(scales*np.abs(self.guide._sample_mantegna_levy(n)), .05, .95)
                mask = self.rng.random(n) < probability
                candidate[mask] = best.best_targets[mask]
            else:
                donor = population[int(self.rng.integers(len(population)))]
                mask = self.rng.random(n) < .5
                candidate[mask] = donor[mask]
            if cfg.strategy != "pso":
                j = int(self.rng.integers(n))
                candidate[j] = (0 if candidate[j]!=0 else home[j]) if binary else (candidate[j] + int(self.rng.integers(1, choices))) % choices
            proposals += 1
            changed_users += int(np.count_nonzero(candidate!=population[i]))
            score = evaluate(candidate)
            if score < personal_scores[i]:
                personal[i], personal_scores[i] = candidate.copy(), score
            if cfg.strategy == "pso" or score < scores[i]:
                population[i], scores[i] = candidate, score
            record_search(previous_profiles)
        raw_search_best_cost = best.best_cost
        if cfg.deadline_aware:
            best = deployment(time.perf_counter()-started)[0]
            curve.append(best.best_cost)
        if not search:
            stop_reason = 'anchor_only' if cfg.strategy=='anchor' else 'event_skip'
        elif stalled>=cfg.patience:
            stop_reason = 'patience'
        self.previous_targets, self.previous_context = best.best_targets.copy(), context
        elapsed = time.perf_counter()-started
        self.last_diagnostics = dict(evaluations=evaluations, search_triggered=bool(search),
                                     budget_ms=cfg.budget_ms, budget_overrun=elapsed*1000 > cfg.budget_ms,
                                     greedy_cost=greedy_cost, raw_greedy_improvement=greedy_cost-best.best_cost)
        self.last_diagnostics.update(anchor_cost=anchor_cost, search_gain=anchor_cost-best.best_cost,
                                     proposals=proposals, cache_hits=cache_hits,
                                     search_improvements=improvements-anchor_improvements,
                                     mean_changed_users=changed_users/max(1,proposals),action_scope=cfg.action_scope)
        self.last_diagnostics.update(deadline_aware=cfg.deadline_aware,stop_reason=stop_reason,
                                     stopping_enabled=bool(cfg.deadline_aware and cfg.stopping_enabled),
                                     raw_search_best_cost=raw_search_best_cost,
                                     **(controller.last if controller is not None else {}))
        return OptimizationOutput(best.best_cost, best.best_targets.copy(), best.best_cpu.copy(), best.best_bw.copy(),
                                  curve, elapsed, evaluations, best.evaluation_result, "Anytime-WE-KB-FPA")
