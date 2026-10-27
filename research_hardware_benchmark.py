"""Small development workload for serial/parallel throughput; not paper results."""
from concurrent.futures import ProcessPoolExecutor
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from research_parallel import worker
from experiments.run_research import Scenario


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    settings=dict(evaluations=620,population=20,slots=4,shock_slots=4,include_drl=False,checkpoint_every=1,reference_seconds=.1,mip_gap=.01)
    scenario=Scenario('channel','throughput-development',users=20,edges=4,sinr_db=10.,q=0)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        serial_jobs=[(settings,str(root/'serial'),asdict(scenario),910001+i,42,str(root),0) for i in range(8)]
        parallel_jobs=[(settings,str(root/'parallel'),asdict(scenario),910001+i,42,str(root),0) for i in range(8)]
        tick=time.perf_counter()
        for job in serial_jobs:
            worker(job)
        serial=time.perf_counter()-tick;tick=time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
            list(pool.map(worker,parallel_jobs))
        parallel=time.perf_counter()-tick
    result=dict(development_cases=8,workers=args.workers,serial_seconds=serial,parallel_seconds=parallel,
        observed_throughput_speedup=serial/parallel,logical_processors=os.cpu_count(),
        scope='Development throughput probe only; final timing and QoS data are not taken from this benchmark.')
    Path('hardware_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
