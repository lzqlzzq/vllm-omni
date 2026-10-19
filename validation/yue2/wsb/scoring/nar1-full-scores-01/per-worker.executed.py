def run(root, metric, input_path):
    if metric != 'songbench': configure(root)
    data = json.loads(Path(input_path).read_text()); config = data['config']; jobs = data['jobs']
    import torch
    torch.manual_seed(config['seed'])
    print(json.dumps({'metric': metric, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}), flush=True)
    factories = {'mulan': MuLanScorer, 'allmusiccaps': AllMusicCapsScorer, 'per': PERScorer}
    if metric == 'q3o':
        from wsb_q3o import PaperQ3OScorer
        factories['q3o'] = PaperQ3OScorer
    if metric == 'songbench':
        from .evaluation import SongBench
        scorer = SongBench(root)
    else:
        scorer = factories[metric](root, config)
    failures = 0
    if metric == 'per':
        batch_size = config['per']['asr_batch_size']
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start+batch_size]
            started = time.perf_counter()
            try:
                for row in batch:
                    if digest(row['audio_path']) != row['audio_sha256']:
                        raise ValueError('Audio changed after manifest preparation')
                results = scorer.score_batch(batch)
            except Exception as exc:
                # Do not silently change batch geometry or omit failed songs.
                results = [{'status':'failed','error':f'{type(exc).__name__}: {exc}',
                            'traceback':traceback.format_exc(),'raw_decode':scorer.raw} for _ in batch]
            elapsed = time.perf_counter()-started
            for row, result in zip(batch, results):
                result.update(batch_seconds=elapsed, batch_size=len(batch))
                failures += result['status'] == 'failed'
                save_result(Path(row['result_path']), row['identity'], result)
                print(f'per {row["id"]}: {result["status"]}', flush=True)
        return 1 if failures else 0
    for index, row in enumerate(jobs):
        start = time.perf_counter()
        try:
            if digest(row['audio_path']) != row['audio_sha256']:
                raise ValueError('Audio changed after manifest preparation')
            if metric == 'songbench':
                scores = scorer.score(Path(row['audio_path']))
                result = {'status': 'complete', 'scores': scores['songbench_raw'], **scores}
                result['scores']['Global Avg'] = sum(scores['songbench_raw'].values())/7
            else:
                result = scorer.score(row)
            object_hash(result)  # Reject NaN / infinity before recording success.
        except Exception as exc:
            result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
            if hasattr(scorer, 'capture'): result['raw_decode'] = scorer.capture
            if hasattr(scorer, 'raw'): result['raw_decode'] = scorer.raw
        torch.cuda.synchronize()
        result['seconds'] = time.perf_counter()-start
        failures += result['status'] == 'failed'
        save_result(Path(row['result_path']), row['identity'], result)
        print(f'{metric} {index+1}/{len(jobs)} {row["id"]}: {result["status"]}', flush=True)
    return 1 if failures else 0
