"""Atomic full-agent exports. Load only trusted, locally generated pickle files."""
import hashlib
import os
from pathlib import Path
import pickle
import uuid


def source_hashes():
    root = Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['dqn.py','droo.py','neural.py','allocation.py','cost_engine.py','config.py',
             'training.py','evaluation.py','environment_model.py','channel_model.py','task_model.py']}


def save_learner_state(agent, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = pickle.dumps(dict(agent=agent,metadata=metadata,sources=source_hashes()),protocol=pickle.HIGHEST_PROTOCOL)
    envelope = dict(schema=1,sha256=hashlib.sha256(payload).hexdigest(),payload=payload)
    temporary = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(envelope,stream,protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def load_learner_state(path):
    envelope = pickle.loads(Path(path).read_bytes())
    if envelope.get('schema')!=1 or hashlib.sha256(envelope['payload']).hexdigest()!=envelope['sha256']:
        raise ValueError('Invalid full-learner export or checksum')
    data = pickle.loads(envelope['payload'])
    if data['sources']!=source_hashes():
        raise ValueError('Learner implementation changed; retrain rather than silently reinterpret saved state')
    return data['agent'],data['metadata']
