"""Freeze both sources and common deployment settings before the sequential paired run."""
import copy
import hashlib
import json
import os
from importlib.metadata import distributions
import subprocess
from pathlib import Path
import yaml

root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
counts={}
for arm,directory in [('upstream','upstream'),('candidate','source')]:
    base=root/directory
    actual={str(p.relative_to(base)):sha(p) for p in sorted((base/'vllm_omni').rglob('*')) if p.suffix in ['.py','.yaml'] and p.name!='_version.py'}
    assert actual==json.loads((root/f'evidence/{arm}-source.json').read_text()),arm
    counts[arm]=len(actual)
configs={a:yaml.safe_load((root/f'configs/{a}.yaml').read_text()) for a in ['upstream','candidate']}
normalized={}
for arm,config in configs.items():
    normalized[arm]=copy.deepcopy(config)
    normalized[arm].pop('pipeline')
    stage=normalized[arm]['stages'][0];stage.pop('worker_cls',None)
    stage['default_sampling_params']['extra_args'].pop('yue2_max_hold_steps')
    stage['default_sampling_params'].pop('max_tokens')
    hf=stage['hf_overrides'];hf.pop('architectures')
    for key in ['yue2_nar_batch_wait_ms','yue2_nar_batch_max_tokens','yue2_nar_batch_size','yue2_vae_batch_size']:hf.pop(key,None)
assert normalized['upstream']==normalized['candidate']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
runtime={str(p.relative_to(root)):sha(p) for d in ['scripts','configs'] for p in sorted((root/d).glob('*')) if p.is_file()}
plan=dict(order=['upstream-01','candidate-01','candidate-02','upstream-02'],requests=32,warmups=2,concurrency=2,offset=1,max_abc=4096,max_frames=9000,mps='off',omp_mkl=1,upstream='f9143976b688cf1e78796ee080d7294f6e019960 + shared HF/VAE loading fix 631f5573',candidate='Rebased unified main path with accelerator capture preparation and delayed coalesced-token flush; automatic ABC, CFG=1, NAR1/VAE1/window0/token-budget0; no packed NAR benefit claimed',source_counts=counts,common_config=normalized['upstream'],runtime_files=runtime,save_audio_latent=False,observers=False)
(root/'evidence/plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(root/'evidence/preflight.json').write_text(json.dumps(dict(complete=True,source_counts=counts,configs_matched=True,gpu_empty=True),indent=2)+'\n')
print('PREFLIGHT_COMPLETE',counts,flush=True)

for slots in [1,2]:
    result=json.loads((root/f'artifacts/gpu-{slots}.json').read_text())
    assert result['complete'] and result['slots']==slots
    assert result['state']['states']==0 and not result['state']['active'] and not result['state']['waiting']
    assert (result['state']['cfg_steps']>0)==(slots==2)
http=json.loads((root/'artifacts/http/result.json').read_text())
assert all(http[k]['status']==200 for k in ['off','guided-off','melody-auto','full-auto','external'])
assert all(http[k]['status']==400 for k in ['cfg-rejected','budget-rejected'])
assert http['server_exit_code']==0
assets=json.loads((root/'evidence/assets.json').read_text())
for asset in assets:
    h=hashlib.sha256()
    with (Path(asset['path'])/'model.safetensors').open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==asset['sha256'],asset['repo']
assert os.environ.get('VLLM_OMNI_EVENT_DRIVEN_ORCH','1').lower() in ['1','true','yes','on']
assert not os.environ.get('PYTORCH_ALLOC_CONF') and not os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
(root/'evidence/environment.json').write_text(json.dumps(dict(
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit','--format=csv,noheader'],text=True),
    packages=sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()),
    env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','VLLM_OMNI_EVENT_DRIVEN_ORCH','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF']},
    assets_sha256_verified=True,abc_cfg_http_verified=True),indent=2)+'\n')
print('ASSETS_AND_FUNCTIONAL_GATE_COMPLETE',flush=True)
