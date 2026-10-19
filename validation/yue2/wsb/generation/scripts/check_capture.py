"""Exercise the actual capture hook with CPU stand-ins, without GPU dependencies."""
import ast
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace as S

# Execute only the external hook and ID validator; production imports need CUDA.
tree = ast.parse(Path(__file__).with_name('generate_wsb.py').read_text())
worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArtifactWorker')
method = worker.body[0]
method.body = [n for n in method.body if not isinstance(n, ast.ImportFrom)]
external = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'external_id')
class Tensor:
    def __init__(self, frames): self.ndim=2; self.shape=(frames,64)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return [self.shape[0]]
class Job:
    def release(self): self.keep.clear(); self._released=True
class NP:
    @staticmethod
    def isfinite(value): return S(all=lambda: True)
    @staticmethod
    def save(path,value): Path(path).write_text(json.dumps(value))
ns=dict(Path=Path,os=os,json=json,re=re,torch=S(Tensor=Tensor),np=NP,_SynthesisJob=Job,batched_synthesis=S(BatchedCachedNAR=lambda *a, **k: S(lengths=[103,104],nar_length=207,key_offsets=[0,100,200])))
exec(compile(ast.Module(body=[method,external],type_ignores=[]),'capture','exec'),ns)
original=Job.release
with tempfile.TemporaryDirectory() as folder:
    os.environ['WSB_OUTPUT']=folder
    queue=S()
    worker=S(model_runner=S(model=S(sample=S(__self__=S(_synthesis=queue)))))
    assert ns['capture_latents'](worker)
    for i,failed,cancelled in [(0,False,False),(1,True,False),(2,False,True)]:
        job=Job(); job.__dict__.update(_released=False,keep=[Tensor(100+i)],request_id=f'wsb_{i:03}_r0_s831001-abcdef01',cancelled=cancelled,error=RuntimeError() if failed else None,finite=True,t_start=1,t_done=2)
        job.release(); job.release()
        assert not job.keep
        assert (Path(folder)/'captures'/f'wsb_{i:03}_r0_s831001.npy').exists() == (not failed and not cancelled)
    queue.members=[S(request_id=f'wsb_{i:03}_r0_s831001') for i in (3,4)]
    job=Job(); job.__dict__.update(_released=False,keep=[Tensor(103),Tensor(104)],cancelled=False,error=None,finite=True,t_start=1,t_done=2)
    job.release()
    assert json.loads((Path(folder)/'captures/wsb_004_r0_s831001.npy').read_text())==[104]
    batches=[json.loads(l) for l in (Path(folder)/'captures/batches.jsonl').read_text().splitlines()]
    assert [b['size'] for b in batches]==[1,2]
    engine=ns['batched_synthesis'].BatchedCachedNAR()
    record=json.loads((Path(folder)/'captures/packed-nar.jsonl').read_text())
    assert record['frames']==[103,104] and len(record['ids'])==2 and engine.nar_length==207
    for bad in ['../wsb_000_r0_s831001','wsb_000_r0_s831001-invalid']:
        try: ns['external_id'](bad)
        except ValueError: pass
        else: raise AssertionError(bad)
Job.release=original
print('PASS: B1/B2 identity, native release, duplicate release, failure, cancellation, unsafe ID')
