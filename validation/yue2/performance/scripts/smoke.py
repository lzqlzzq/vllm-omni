import asyncio
import json
import os
from pathlib import Path

import torch
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.model_executor.models.yue2.request import prepare_song
from vllm_omni.model_executor.models.yue2.yue2 import CODEC_OFFSET, CODEC_SIZE
from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT/'evidence/assets.json').read_text())

class Probe:
    def install_probe(self):
        model = self.model_runner.model
        model._unified_cfg_steps = 0
        sample = model.sample
        def observe(logits, metadata):
            pairs = [(i, a['yue2_cfg_parent']) for i,a in enumerate(model._cfg_step) if a and a.get('yue2_cfg_parent')]
            result = sample(logits, metadata)
            rows = {rid:i for i,(rid,_,_) in enumerate(model._step_rows)}
            for neg,parent in pairs:
                assert torch.equal(result.sampled_token_ids[neg], result.sampled_token_ids[rows[parent]])
            if pairs:
                model._unified_cfg_steps += 1
            return result
        model.sample = observe
        return type(getattr(model, 'runnable', model)).__name__

    def state(self):
        model = self.model_runner.model
        return dict(states=len(model._states), active=bool(model._synthesis.active),
                    waiting=len(model._synthesis.waiting), cfg_steps=model._unified_cfg_steps)

async def main():
    slots = int(os.environ['SMOKE_SLOTS'])
    model,vae = ASSETS[:2]
    tokenizer = YuE2TextTokenizer.from_pretrained(model['path'])
    engine = AsyncOmni(model=model['path'], revision=model['revision'],
        worker_extension_cls='smoke.Probe',
        stage_overrides={'0': dict(max_num_seqs=slots, max_model_len=4096,
            max_num_batched_tokens=8192, kv_cache_memory_bytes=3*1024**3,
            hf_overrides=dict(yue2_vae_path=vae['path'],yue2_vae_revision=vae['revision']))})
    records=[]
    async def song(name, scale=1.0, cot='off', abc=None):
        prompt,params=prepare_song(tokenizer.encode,'Warm acoustic folk, gentle guitar',
            '[Verse]\nMorning light across the bay\nCarry all my dreams away\n[Outro]',
            cot=cot,abc=abc,seed=831001,max_frames=200,max_abc_tokens=32,max_model_len=4096,cfg_scale=scale)
        result=None
        try:
            async for result in engine.generate(prompt,request_id=name,sampling_params_list=[params]):
                pass
        except RuntimeError as error:
            if name != 'reject-cfg' or 'YuE2 CFG requires' not in str(error):
                raise
            records.append(dict(name=name,rejected=True,message=str(error)))
            return
        assert result is not None
        mm=result.multimodal_output or {}
        audio=mm.get('audio')
        if audio is None:
            finish=result.outputs[0].finish_reason
            assert name=='reject-cfg' and str(finish)=='error', (name,finish,result)
            records.append(dict(name=name,rejected=True,finish=str(finish)))
            return
        assert name!='reject-cfg'
        tokens = result.outputs[0].token_ids
        assert sum(CODEC_OFFSET <= t < CODEC_OFFSET+CODEC_SIZE for t in tokens)==200, (name,len(tokens))
        assert tuple(audio.shape)==(2,200*1920-64), (name,audio.shape)
        assert torch.isfinite(audio).all() and audio.abs().max()>0
        if cot!='off' and abc is None:
            assert len(mm['abc_token_ids'])>0
        meta=mm.get('meta') or {k[5:]:v for k,v in mm.items() if k.startswith('meta.')}
        assert not torch.as_tensor(meta['error']).item()
        records.append(dict(name=name,shape=list(audio.shape),abc_tokens=len(mm.get('abc_token_ids',[]))))
        print(json.dumps(records[-1]),flush=True)
    try:
        print('MODEL',await engine.collective_rpc('install_probe'),flush=True)
        await song('ordinary')
        await song('automatic',cot='full')
        await song('external',cot='melody',abc='X:1\nT:Test\nM:4/4\nL:1/8\nK:C\nCDEF GABc|')
        if slots==1:
            await song('reject-cfg',scale=1.5)
            await song('after-rejection')
        else:
            await asyncio.gather(song('guided',scale=1.5),song('guided-auto',scale=1.5,cot='full'),song('mixed-plain'))
        state=(await engine.collective_rpc('state'))[0][0]
        assert state['states']==0 and not state['active'] and not state['waiting'],state
        assert (state['cfg_steps']>0)==(slots>1),state
        (ROOT/f'artifacts/gpu-{slots}.json').write_text(json.dumps(dict(complete=True,slots=slots,records=records,state=state),indent=2)+'\n')
        print('COMPLETE',slots,state,flush=True)
    finally:
        engine.close()

if __name__=='__main__':
    asyncio.run(asyncio.wait_for(main(), timeout=900))
