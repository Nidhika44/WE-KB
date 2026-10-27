"""Finish quality reporting after an older launcher exits; computations are untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from quality_reports import create_quality_reports


def finalize(output):
    output=Path(output)
    protocol=json.loads((output/'parallel_protocol.json').read_text());settings=protocol['settings']
    frame=pd.read_csv(output/'quality_results/quality_slots.csv',dtype={'feasible_task_ids':'string'})
    expected=len(settings['seeds'])*len(settings['scenarios'])
    count=len(frame[['study','scenario','seed']].drop_duplicates())
    if count!=expected or frame.duplicated(['study','scenario','seed','slot','algorithm']).any():
        raise ValueError('Quality data is incomplete or contains duplicate records')
    create_quality_reports(frame,output/'quality_results',settings)
    (output/'quality_report_revision.json').write_text(json.dumps(dict(change='Quality-only reporter avoids undefined timing crossovers; algorithm data unchanged',
        finalizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=count),indent=2))
    (output/'quality_complete.json').write_text(json.dumps(dict(total=count,reporting_repaired=True),indent=2))
    print(f'Quality reporting complete: {count} cases',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='transfer/parallel_option')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--pid-file',default='quality_pid.json')
    args=parser.parse_args()
    if args.wait:
        pid=json.loads(Path(args.pid_file).read_text())['pid']
        while True:
            process=Path(f'/proc/{pid}')
            if not process.exists():
                break
            try:
                stat=(process/'stat').read_text()
                command=(process/'cmdline').read_bytes()
            except FileNotFoundError:
                break
            if b'research_parallel.py' not in command:
                break
            if stat.split(') ',1)[1].startswith('Z'):
                break
            time.sleep(20)
    finalize(args.output_dir)


if __name__=='__main__':
    main()
