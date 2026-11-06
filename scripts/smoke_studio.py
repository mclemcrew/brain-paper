"""Real local integration smoke; synthetic tones verify plumbing, not musical quality.

Requires the selected model already installed in --root/models. Does not download.
Writes a machine-readable report with actual hardware and job measurements.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import tempfile
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ravi.local.store import Store, atomic_json
from ravi.local.jobs import Jobs
from ravi.local.catalog import hardware

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=ROOT/'.local-studio')
parser.add_argument('--model', choices=['sa3-medium','sa3-small'], default='sa3-medium')
parser.add_argument('--profile', choices=['8gb','16gb','32gb'], default='16gb')
parser.add_argument('--steps',type=int,default=2)
parser.add_argument('--worker-binary',type=Path,help='Test the frozen worker and its bundled MLX source')
args=parser.parse_args()
store=Store(args.root);p=store.create('Integration smoke · synthetic tones')
p=store.update(p['id'], {'model_id': args.model, 'profile': args.profile})
with tempfile.TemporaryDirectory(dir=args.root) as tmp:
    for i in range(3):
        path=Path(tmp)/f'Piece {i}.wav'
        t=np.arange(44100*3)/44100
        y=(.08*np.sin(2*np.pi*(220+i*55)*t)).astype('float32')
        sf.write(path,y,44100,subtype='FLOAT');store.import_file(p['id'],path)
p=store.split(p['id']);p=store.update(p['id'],{'description':'A sustained quiet synthesizer tone'})
for t in p['tracks']:
    p=store.update(p['id'],{'track':{'id':t['id'],'caption':'A sustained quiet synthesizer tone','reviewed':True}})
binary=args.worker_binary.resolve() if args.worker_binary else None
runtime=binary.parent/'_internal/sa3-mlx' if binary else ROOT/'.local-runtime/stable-audio-3/optimized/mlx'
jobs=Jobs(store,runtime,worker_binary=binary)
report={'hardware':hardware(),'packaged':bool(binary),'purpose':'integration only; synthetic tones are not a quality evaluation','jobs':[]}
def run(kind,**extra):
    job=jobs.submit({'kind':kind,'project_id':p['id'],'prompt':'Quiet solo piano, slow lyrical notes','seconds':3,'steps':args.steps,**extra})
    while jobs.active: time.sleep(.5)
    result=next(j for j in jobs.list() if j['id']==job['id'])
    report['jobs'].append(result);atomic_json(args.root/'smoke-report.json',report)
    print(kind,result['status'],result['elapsed_seconds'],result.get('peak_rss_gb'),flush=True)
    if result['status']!='completed':raise RuntimeError(result['message'])
    return result
try:
    run('generate')
    run('train')
    cp=jobs.checkpoints(p['id'])[0]
    run('generate',adapter_id=cp['id'])
    run('edit',source_id=p['tracks'][0]['id'],start=.5,end=2.5,adapter_id=cp['id'])
    run('transcribe',source_id=p['tracks'][0]['id'])
finally:jobs.close()
print('Report:',args.root/'smoke-report.json')
