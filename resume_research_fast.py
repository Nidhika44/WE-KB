"""Resume the frozen research suite with batched result-file writes.

Algorithm modules and checkpoint identities are unchanged. This launcher records
its own source hash separately and preserves the original per-slot checkpoints.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

for _thread_variable in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[_thread_variable]='1'

import pandas as pd
from we_kb_fpa.checkpoint import Checkpoint,RunLock,identity_hash,implementation_signature
from experiments.run_research import Scenario,run_case
from experiments.research_report import report


def read_protocol(output):
    frozen=json.loads((output/'frozen_protocol.json').read_text())
    config=json.loads((output/'config.json').read_text())
    if config!=frozen:
        raise ValueError('Saved configuration differs from the frozen protocol')
    if frozen['implementation']!=implementation_signature():
        raise ValueError('Frozen algorithm code or dependency environment changed')
    settings=frozen['settings']
    hardware=settings.get('hardware',{})
    if hardware.get('processor',platform.processor())!=platform.processor():
        raise ValueError('Processor differs from the frozen timing environment')
    for variable,value in hardware.get('threads',{}).items():
        if os.environ.get(variable)!=value:
            raise ValueError(f'Thread environment differs: {variable}')
    for name,expected in settings.get('learner_checksums',{}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen learner input changed: {name}')
    root=output/'checkpoints'/identity_hash(settings)[:24]
    if not root.is_dir():
        raise ValueError('Matching checkpoint directory is missing; no new namespace will be created')
    return settings,root


def cases_for(settings):
    scenarios=[]
    for item in settings['scenarios']:
        item=dict(item);item['affected_edges']=tuple(item['affected_edges'])
        scenarios.append(Scenario(**item))
    return [(seed,scenario,42+index) for index,seed in enumerate(settings['seeds']) for scenario in scenarios]


def checkpoint_for(settings,root,scenario,seed):
    from dataclasses import asdict
    return Checkpoint(root,f'{scenario.study}-{scenario.name}-{seed}',
                      dict(scenario=asdict(scenario),seed=seed,settings=settings))


def write_aggregate(output,records):
    pending=output/'per_seed_slots.pending.csv'
    pd.DataFrame(records).to_csv(pending,index=False)
    pending.replace(output/'per_seed_slots.csv')


def run(output,aggregate_every=25,check_only=False):
    output=Path(output)
    if aggregate_every<1:
        raise ValueError('Aggregate write interval must be positive')
    if check_only:
        settings,root=read_protocol(output)
        print(f'Frozen code, learner files, environment and checkpoint namespace match. {len(cases_for(settings))} cases.',flush=True)
        return
    try:
        lock=RunLock(output/'run.lock')
        lock.__enter__()
    except RuntimeError as error:
        raise RuntimeError('The original run is still active. Press Ctrl+C in its terminal before using this launcher.') from error
    started=time.perf_counter();computed=0;restored=0;records=[]
    try:
        settings,root=read_protocol(output)
        cases=cases_for(settings)
        history=dict(started_utc=datetime.now(timezone.utc).isoformat(),launcher='resume_research_fast.py',
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            frozen_protocol_sha256=hashlib.sha256((output/'frozen_protocol.json').read_bytes()).hexdigest(),
            aggregate_every_new_cases=aggregate_every,
            change='Result I/O only: skip cached CSV rewrites, batch aggregate CSV; original run_case and per-slot checkpoints retained')
        with (output/'fast_resume_history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(history)+'\n')
        seed_directory=output/'per_seed';seed_directory.mkdir(exist_ok=True)
        pending=[]
        # Load each completed payload once; never rebuild the large CSV per cached case.
        for seed,scenario,model_seed in cases:
            checkpoint=checkpoint_for(settings,root,scenario,seed)
            state=checkpoint.load()
            if state and state.get('complete'):
                rows=state['records'];records.extend(rows);restored+=1
                path=seed_directory/f'{scenario.study}-{scenario.name}-{seed}.csv'
                if not path.exists():
                    pd.DataFrame(rows).to_csv(path,index=False)
            else:
                pending.append((seed,scenario,model_seed))
        print(f'Restored {restored}/{len(cases)} completed cases in {time.perf_counter()-started:.1f} s; {len(pending)} remain.',flush=True)
        write_aggregate(output,records)
        last_export=time.perf_counter()
        for seed,scenario,model_seed in pending:
            rows=run_case(settings,root,scenario,seed,model_seed)
            records.extend(rows);computed+=1
            pd.DataFrame(rows).to_csv(seed_directory/f'{scenario.study}-{scenario.name}-{seed}.csv',index=False)
            if computed%aggregate_every==0 or time.perf_counter()-last_export>=300:
                write_aggregate(output,records);last_export=time.perf_counter()
            print(f'Completed {restored+computed}/{len(cases)}: {scenario.study}/{scenario.name}, seed {seed}; '
                  f'newly computed {computed}; elapsed {(time.perf_counter()-started)/60:.1f} min',flush=True)
        write_aggregate(output,records)
        report(pd.DataFrame(records),output,settings)
        timing=dict(seconds_this_resume=time.perf_counter()-started,restored_cases=restored,new_cases=computed,
                    total_cases=len(cases),note='Prior-session wall time is not included')
        (output/'fast_resume_timing.json').write_text(json.dumps(timing,indent=2))
        (output/'timing.json').write_text(json.dumps(dict(seconds=timing['seconds_this_resume'],cases=len(cases),
            restored_cases=restored,new_cases=computed,note=timing['note']),indent=2))
        print(f'Research suite complete: {output.resolve()}',flush=True)
    finally:
        lock.__exit__(None,None,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='output/research_paper')
    parser.add_argument('--aggregate-every',type=int,default=25)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    run(args.output_dir,args.aggregate_every,args.check_only)


if __name__=='__main__':
    main()
