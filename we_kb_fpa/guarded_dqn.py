"""Explicit frozen-DQN plus common-anchor control; pure DQN remains separate."""
import copy
from dataclasses import replace
import time
from we_kb_fpa.anytime import AnytimeConfig,AnytimeWEKBFPA
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase


class AnchoredFrozenDQN:
    def __init__(self,agent,budget_ms,net_config,seed):
        self.agent=copy.copy(agent)
        self.budget_ms=budget_ms
        self.anchor=AnytimeWEKBFPA(AnytimeConfig(strategy='anchor',budget_ms=budget_ms,event_trigger=False),net_config=net_config,seed=seed)
        self.knowledge=WirelessEdgeKnowledgeBase(net_config=net_config)
        self.last_diagnostics={}

    def run(self,*args,training=False):
        if training:
            raise ValueError('This explicitly labelled control freezes the neural policy')
        started=time.perf_counter()
        prediction=self.agent.run(*args,training=False)
        remaining=max(.000001,self.budget_ms-(time.perf_counter()-started)*1000)
        self.anchor.config=replace(self.anchor.config,budget_ms=remaining)
        anchored=self.anchor.optimize(*args,knowledge_base=self.knowledge)
        best=min([prediction,anchored],key=lambda output:output.best_cost)
        self.anchor.previous_targets=best.best_targets.copy()
        self.last_decision_seconds=time.perf_counter()-started
        self.knowledge.add_experience(args[4],args[0],args[5],best.best_targets,best.best_cpu,best.best_bw,best.best_cost)
        elapsed=time.perf_counter()-started
        self.last_learning_seconds=elapsed-self.last_decision_seconds
        self.last_diagnostics=dict(budget_ms=self.budget_ms,budget_overrun=elapsed*1000>self.budget_ms,
                                   greedy_cost=self.anchor.last_diagnostics['greedy_cost'],
                                   raw_greedy_improvement=self.anchor.last_diagnostics['greedy_cost']-best.best_cost)
        best.wall_clock_time_sec=elapsed
        best.algorithm_name='DQN-Frozen-AnchorSafe'
        return best
