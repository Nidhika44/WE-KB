"""Parallel quality evaluation with a separate, isolated serial timing benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import time
import zipfile

for _variable in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[_variable]='1'

import numpy as np
import pandas as pd
from resume_research_fast import read_protocol,cases_for,checkpoint_for
from we_kb_fpa.checkpoint import Checkpoint,RunLock,identity_hash,implementation_signature
from experiments.run_research import run_case,Scenario
from experiments.research_report import report,plot_sweep

TIMING_FIELDS={'runtime_ms','retrieval_ms','kb_build_seconds','delay_adjusted_violation_pct',
    'delay_feasible_cost','delay_physical_violation_lower_bound_pct','reference_runtime_sec'}
PACKAGES=['numpy','scipy','pandas','matplotlib','scikit-learn']


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2));pending.replace(path)


def quality_frame(rows,mode):
    frame=pd.DataFrame(rows).copy()
    if 'execution_mode' not in frame:
        frame['execution_mode']=mode
    for column in list(frame.columns):
        if column in TIMING_FIELDS or 'wall_clock' in column or column.endswith('_seconds'):
            frame[column]=np.nan
    for column in TIMING_FIELDS:
        frame[column]=np.nan
    return frame


def context(base,portable=False):
    base=Path(base)
    if not portable:
        settings,root=read_protocol(base)
        return settings,root,None
    transfer=json.loads((base/'quality_transfer.json').read_text())
    project=Path(__file__).resolve().parent
    for name,expected in transfer['source_sha256'].items():
        path=project/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Transferred source changed: {name}')
    for name,expected in transfer['packages'].items():
        if version(name)!=expected:
            raise ValueError(f'Use the transferred requirements: {name} must be {expected}')
    settings=copy.deepcopy(transfer['settings'])
    settings['models_dir']=str((base/'models').resolve())
    # Windows paths need Windows basename handling when the transfer is opened on Linux.
    settings['learner_checksums']={str(base/'models'/name.replace('\\','/').rsplit('/',1)[-1]):value
                                    for name,value in transfer['settings'].get('learner_checksums',{}).items()}
    for name,expected in settings['learner_checksums'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Transferred learner checksum mismatch')
    settings['hardware']=dict(processor=platform.processor(),python=platform.python_version(),
        threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']})
    return settings,None,transfer


def case_key(seed,scenario):
    return f'{scenario.study}-{scenario.name}-{seed}'


def worker(job):
    settings,root,scenario_dict,seed,model_seed,directory,prefix=job
    scenario=Scenario(**scenario_dict)
    started=time.perf_counter()
    rows=run_case(settings,Path(root),scenario,seed,model_seed)
    frame=pd.DataFrame(rows)
    frame['execution_mode']=np.where(frame.slot<prefix,'original_serial_prefix','parallel_quality')
    frame['worker_pid']=os.getpid()
    path=Path(directory)/f'{case_key(seed,scenario)}.csv'
    pending=path.with_suffix('.pending.csv');frame.to_csv(pending,index=False);pending.replace(path)
    return str(path),time.perf_counter()-started


def export_quality(base,destination):
    base=Path(base);settings,root=read_protocol(base)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parent
    sources=[p for folder in ['we_kb_fpa','experiments'] for p in (source_root/folder).glob('*.py')]
    sources += [source_root/name for name in ['research_parallel.py','research_serial_benchmark.py','resume_research_fast.py',
                'research_hardware_benchmark.py','quality_reports.py','quality_report_finalize.py']]
    hashes={p.relative_to(source_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    pieces=[];completed=[];probe=None
    for seed,scenario,model_seed in cases_for(settings):
        state=checkpoint_for(settings,root,scenario,seed).load()
        if state and state.get('complete'):
            frame=quality_frame(state['records'],'original_serial')
            pieces.append(frame);completed.append(case_key(seed,scenario))
            if probe is None and scenario.study=='scalability':
                probe=dict(seed=seed,scenario=asdict(scenario),model_seed=model_seed,
                    expected=frame[frame.slot==0][['algorithm','search_objective','feasible_cost','violation_pct','feasible_task_ids']].to_dict('records'))
    transfer=dict(schema=1,settings=settings,completed_cases=completed,source_sha256=hashes,
        parent_frozen_sha256=hashlib.sha256((base/'frozen_protocol.json').read_bytes()).hexdigest(),
        packages={name:version(name) for name in PACKAGES},probe=probe,
        scope='Quality-only cross-platform continuation. Timing and delay-adjusted results must not be pooled. No OCI credentials included.')
    pending=destination.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path,path.relative_to(source_root).as_posix())
        archive.write(source_root/'requirements.txt','requirements.txt')
        archive.writestr('transfer/quality_transfer.json',json.dumps(transfer,indent=2))
        if pieces:
            archive.writestr('transfer/completed_quality.csv',pd.concat(pieces,ignore_index=True).to_csv(index=False))
        for name in settings.get('learner_checksums',{}):
            archive.write(name,'transfer/models/'+Path(name).name)
    pending.replace(destination)
    print(f'Exported {len(completed)} completed cases to {destination.resolve()}; credentials and private keys excluded.',flush=True)
    return transfer


def validate_probe(settings,transfer,output):
    if not transfer or not transfer.get('probe') or (output/'equivalence_probe.json').exists():
        return
    probe=transfer['probe'];config=copy.deepcopy(settings);config['slots']=2
    scenario=Scenario(**probe['scenario'])
    rows=run_case(config,output/'probe_checkpoints',scenario,probe['seed'],probe['model_seed'])
    actual={row['algorithm']:row for row in rows if row['slot']==0}
    for expected in probe['expected']:
        row=actual[expected['algorithm']]
        for metric in ['search_objective','feasible_cost','violation_pct']:
            if not np.isclose(row[metric],expected[metric],rtol=1e-8,atol=1e-8):
                raise ValueError(f'Cross-platform probe differs: {metric}')
        if str(row['feasible_task_ids'])!=str(expected['feasible_task_ids']):
            raise ValueError('Cross-platform probe feasible task set differs')
    write_json(output/'equivalence_probe.json',dict(passed=True,relative_tolerance=1e-8,
        limitation='One prespecified probe is a portability check, not proof of bitwise equivalence for every case.'))


def produce_quality_report(pieces,output,settings):
    from quality_reports import create_quality_reports
    directory=output/'quality_results';directory.mkdir(exist_ok=True)
    frame=pd.concat(pieces,ignore_index=True)
    create_quality_reports(frame,directory,settings)


def run(base,workers=4,portable=False,skip_serial=False,check_only=False):
    if workers<1 or workers>max(1,os.cpu_count() or 1):
        raise ValueError('Worker count exceeds available logical processors')
    base=Path(base).resolve();settings,original_root,transfer=context(base,portable)
    output=base/'parallel_option'
    if check_only:
        print(f'Validated source and inputs. {len(cases_for(settings))} cases; {workers} workers; timing kept separate.');return
    with RunLock(base/'run.lock'):
        output.mkdir(exist_ok=True)
        protocol=dict(parent_frozen_sha256=transfer['parent_frozen_sha256'] if transfer else hashlib.sha256((base/'frozen_protocol.json').read_bytes()).hexdigest(),
            scope='Parallel quality continuation; existing serial outcomes retained; no mixed timing inference',
            algorithm_implementation=implementation_signature(),launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            serial_harness_sha256=hashlib.sha256(Path(__file__).with_name('research_serial_benchmark.py').read_bytes()).hexdigest(),
            settings=settings,hardware=dict(system=platform.system(),machine=platform.machine(),processors=os.cpu_count()),workers=workers,
            serial_benchmark='Fresh independent controllers at slots 0 and midpoint (exact: slot 0), every scenario and seed; separate dataset.')
        protocol_path=output/'parallel_protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text())!=json.loads(json.dumps(protocol)):
            raise ValueError('Parallel protocol changed; preserve the old output and use a new transfer directory')
        write_json(protocol_path,protocol)
        root=output/'checkpoints'/identity_hash(settings)[:24];root.mkdir(parents=True,exist_ok=True)
        raw=output/'raw_cases';raw.mkdir(exist_ok=True)
        audit=output/'case_audit';audit.mkdir(exist_ok=True)
        pieces=[];jobs=[];restored=0
        transferred=set(transfer['completed_cases']) if transfer else set()
        if transferred:
            inherited=pd.read_csv(base/'completed_quality.csv',dtype={'feasible_task_ids':'string'})
            pieces.append(quality_frame(inherited,'inherited_serial'));restored=len(transferred)
        for seed,scenario,model_seed in cases_for(settings):
            key=case_key(seed,scenario)
            if key in transferred:
                continue
            old=checkpoint_for(settings,original_root,scenario,seed).load() if original_root else None
            if old and old.get('complete'):
                pieces.append(quality_frame(old['records'],'original_serial'));restored+=1;continue
            target=checkpoint_for(settings,root,scenario,seed)
            state=target.load()
            audit_path=audit/f'{key}.json'
            prefix=json.loads(audit_path.read_text())['serial_prefix_slots'] if audit_path.exists() else 0
            if state is None and old:
                prefix=old['next_slot'];target.save(old)
            write_json(audit_path,dict(serial_prefix_slots=prefix))
            if state and state.get('complete'):
                frame=pd.DataFrame(state['records']);frame['execution_mode']=np.where(frame.slot<prefix,'original_serial_prefix','parallel_quality')
                pieces.append(quality_frame(frame,'parallel_quality'));restored+=1
            else:
                jobs.append((settings,str(root),asdict(scenario),seed,model_seed,str(raw),prefix))
        validate_probe(settings,transfer,output)
        started=time.perf_counter();completed=0
        print(f'Retained {restored} complete cases; {len(jobs)} pending; {workers} workers. Partial local checkpoints are retained.',flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            queue=iter(jobs);active={}
            for _ in range(workers):
                job=next(queue,None)
                if job:
                    active[pool.submit(worker,job)]=job
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    job=active.pop(future);path,seconds=future.result()
                    frame=pd.read_csv(path,dtype={'feasible_task_ids':'string'});pieces.append(quality_frame(frame,'parallel_quality'));completed+=1
                    print(f'Completed quality {restored+completed}/{len(cases_for(settings))}; newly computed {completed}; '
                          f'case {job[2]["study"]}/{job[2]["name"]}; elapsed {(time.perf_counter()-started)/60:.1f} min',flush=True)
                    if completed%25==0:
                        directory=output/'quality_results';directory.mkdir(exist_ok=True)
                        pending=directory/'quality_slots.pending.csv';pd.concat(pieces,ignore_index=True).to_csv(pending,index=False);pending.replace(directory/'quality_slots.csv')
                    next_job=next(queue,None)
                    if next_job:
                        active[pool.submit(worker,next_job)]=next_job
        produce_quality_report(pieces,output,settings)
        write_json(output/'quality_complete.json',dict(restored=restored,computed=completed,total=len(cases_for(settings)),seconds=time.perf_counter()-started))
        if not portable and not skip_serial:
            from research_serial_benchmark import run_serial_benchmark
            run_serial_benchmark(settings,output/'serial_benchmark')
        print(f'Quality outputs: {output}. Serial timing is a separate dataset.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='output/research_paper')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--portable-quality',action='store_true')
    parser.add_argument('--skip-serial-benchmark',action='store_true')
    parser.add_argument('--serial-only',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--export-oci')
    args=parser.parse_args()
    if args.export_oci:
        export_quality(args.output_dir,args.export_oci);return
    if args.serial_only:
        if args.portable_quality:
            parser.error('Run the serial timing benchmark on the original PC')
        settings,_=read_protocol(Path(args.output_dir))
        with RunLock(Path(args.output_dir)/'run.lock'):
            from research_serial_benchmark import run_serial_benchmark
            run_serial_benchmark(settings,Path(args.output_dir)/'parallel_option'/'serial_benchmark')
        return
    run(args.output_dir,args.workers,args.portable_quality,args.skip_serial_benchmark,args.check_only)


if __name__=='__main__':
    main()
