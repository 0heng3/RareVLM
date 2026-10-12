"""Continue the frozen scientific experiment with an explicit two-A100 layout."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from zoneinfo import ZoneInfo

from vlm_deep.common import content_sha, read_manifest, sha_file, write_json
from .records import build_index, write_summary, verify


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/vlm_deep_v2.json')
    p.add_argument('--public-output',type=Path)
    a=p.parse_args();repo=PROJECT_ROOT
    c=json.loads(a.config.read_text());migration=a.asset_root/'a100_migration'
    migration.mkdir(parents=True,exist_ok=True)
    original_lock=json.loads((a.run_root/'SOURCE_LOCK.json').read_text())
    if original_lock['config_sha256']!=content_sha(c):raise ValueError('original scientific configuration changed')
    if any(sha_file(SOURCE_ROOT/name)!=digest for name,digest in original_lock['source_sha256'].items()):
        raise ValueError('frozen scientific source changed')
    selection_file=a.run_root/'STAGE2_SELECTION.json';selection=json.loads(selection_file.read_text())
    candidates=[]
    for method in c['methods']:
        scores=[];nll=[]
        for seed in c['training_seeds']:
            d=a.run_root/'stage1'/method/f'seed{seed}'
            e=d/'eval_dev/EVALUATION.json';t=d/'train/TRAIN_RESULT.json'
            recorded=next(row for row in selection['dev_ranking'] if row['method']==method)['input_hashes'][str(seed)]
            if sha_file(e)!=recorded['dev'] or sha_file(t)!=recorded['train']:raise ValueError('stage1 selection inputs changed')
            scores.append(json.loads(e.read_text())['score_percent'])
            nll.append(json.loads(t.read_text())['internal_dev_token_nll'])
        candidates.append((method,statistics.mean(scores),statistics.mean(nll)))
    candidates.sort(key=lambda row:(-row[1],row[2],row[0]))
    if selection['selected_methods']!=[row[0] for row in candidates[:2]] or selection['selection_uses_final_metrics']:
        raise ValueError('frozen dev-only selection changed')
    preflight=migration/'PREFLIGHT_VERIFICATION.json'
    checks=json.loads(preflight.read_text())
    if not checks['passed'] or checks['config_sha256']!=content_sha(c):raise ValueError('A100 preflight is missing or mismatched')
    runtime_sources={str(f.relative_to(SOURCE_ROOT)):sha_file(f) for f in (SOURCE_ROOT/'vlm_a100').glob('*.py')}
    runtime_sources['rarevlm_paths.py']=sha_file(SOURCE_ROOT/'rarevlm_paths.py')
    for path,digest in checks['runtime_source_sha256'].items():
        if sha_file(SOURCE_ROOT/path)!=digest:raise ValueError('preflighted runtime source changed')
    core={'scientific_config_sha256':content_sha(c),'original_source_lock_sha256':sha_file(a.run_root/'SOURCE_LOCK.json'),
          'selection_sha256':sha_file(selection_file),'runtime_source_sha256':runtime_sources,
          'training_layout':{'world_size':2,'microbatch':8,'accumulation':1,'effective_batch':16},
          'evaluation_layout':{'world_size':2,'batch_size':8},'compute':'NF4 language / FP32 computation and vision; TF32 off',
          'preflight_verification_sha256':sha_file(preflight),
          'comparability':'Scientific data/objective/update/selection settings unchanged. Hardware, rank layout, dropout streams and reduction order changed; not bitwise-equivalent.'}
    manifest=migration/'MIGRATION_MANIFEST.json'
    if manifest.exists() and json.loads(manifest.read_text())['contract']!=core:
        raise ValueError('A100 continuation contract changed')
    if not manifest.exists():write_json(manifest,{'created_at':timestamp(),'contract':core})
    phase='a100_continuation_initialization'
    model_root=a.pilot_root/'model'
    shared=['--asset-root',str(a.asset_root),'--model-root',str(model_root),'--config',str(a.config)]
    calibration_path=migration/'CALIBRATION.json'
    calibration=json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    timings_path=migration/'PHASE_TIMINGS.json'
    measurements=json.loads(timings_path.read_text()) if timings_path.exists() else []
    stage2_questions=len(read_manifest(a.asset_root/'prepared/manifest.json')['splits']['train_all'])
    planned_updates=(stage2_questions+c['effective_batch']-1)//c['effective_batch']

    def progress_for(directory,kind,log):
        if kind=='train':
            f=directory/'PROGRESS.json';out=json.loads(f.read_text()) if f.exists() else {}
            entries=[]
            if log.exists():
                for line in log.read_text(errors='replace').splitlines():
                    if not line.startswith('{'):continue
                    try:item=json.loads(line)
                    except json.JSONDecodeError:continue
                    if 'elapsed_active_seconds' in item and 'total_steps' in item:entries.append(item)
            if entries:
                out.update({'latest_step':entries[-1]['step'],'elapsed_active_seconds':entries[-1]['elapsed_active_seconds']})
                usable=[(y['elapsed_active_seconds']-x['elapsed_active_seconds'])/(y['step']-x['step'])
                        for x,y in zip(entries,entries[1:]) if y['step']>x['step'] and x['step']>=25]
                if usable:calibration['training_seconds_per_update']=statistics.median(usable[-5:])
            return out
        count=0
        for f in directory.glob('predictions_rank*.jsonl'):
            with f.open() as h:count+=sum(bool(line.strip()) for line in h)
        return {'completed_questions':count}

    def eta(index,current):
        step_seconds=calibration.get('training_seconds_per_update')
        if step_seconds is None:return {'status':'waiting_for_training_speed_measurement'}
        remaining_train=len(index['missing']['trainings'])*planned_updates
        if phase.endswith('/train'):remaining_train-=current.get('latest_step',current.get('completed_updates',0))
        total=max(remaining_train,0)*step_seconds
        total+=len(index['missing']['trainings'])*calibration.get('dev_nll_and_training_overhead_seconds',240)
        counts={'dev':1617,'textvqa':5000,'docvqa':5349,'mmstar':1500}
        rates=calibration.get('evaluation_questions_per_second',{})
        for item in index['missing']['evaluations']:
            task=item['task'];n=counts.get(task,0)
            expected_phase=f"{item['stage']}/{item['method']}/seed{item['seed']}/eval_{task}"
            if expected_phase==phase:n=max(n-current.get('completed_questions',0),0)
            rate=rates.get(task,rates.get('docvqa'))
            if rate:total+=n/rate+calibration.get('evaluation_load_seconds',70)
        if not rates:return {'status':'training_measured_evaluation_pending','remaining_training_seconds':total}
        return {'status':'provisional_measured_extrapolation','central_seconds':total,
                'range_seconds':[total*.8,total*1.35],
                'assumption':'Continued service, current FP32/NF4 runtime and no failure reruns; per-task rates refine after completion.'}

    def status(name,kind=None,directory=None,log=None):
        index=build_index(a.run_root,c)
        current=progress_for(directory,kind,log) if directory and kind else {}
        state={'phase':name,'status':'running','updated_at':timestamp(),'pipeline_pid':os.getpid(),
               'config_sha256':content_sha(c),'runtime':'vlm_a100','hardware':'2 x NVIDIA A100-SXM4-80GB',
               'effective_batch':16,'completed_training_runs':index['completed_training_runs'],
               'completed_evaluations':index['completed_evaluations'],'progress':current,'eta':eta(index,current)}
        write_json(a.run_root/'STATUS.json',state)
        top=a.asset_root/'EXECUTION_STATUS.json'
        previous=json.loads(top.read_text()) if top.exists() else {}
        previous.update(state);previous.update({'snapshot_at':state['updated_at'],'pipeline_confirmed_alive':True,
            'completed_stage1_trainings':sum(x['stage']=='stage1' for x in index['trainings']),
            'completed_stage2_trainings':sum(x['stage']=='stage2' for x in index['trainings'])})
        write_json(top,previous)
        write_json(calibration_path,calibration)

    def execute(name,module,extra,completed,kind,directory):
        nonlocal phase
        phase=name
        if completed.exists():
            print('reuse completed',name,flush=True);return
        log=a.run_root/(name.replace('/','__')+'_a100.log')
        status(name,kind,directory,log)
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 '-m',module,*shared,*extra]
        print('START',name,flush=True);begun=time.monotonic()
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=repo,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(15);status(name,kind,directory,log)
        if process.returncode:raise RuntimeError(f'{name} failed with exit {process.returncode}; see {log}')
        if not completed.exists():raise ValueError(f'{name} exited without a complete result')
        duration=time.monotonic()-begun
        measurements.append({'phase':name,'wall_seconds':duration,'completed_at':timestamp()})
        write_json(timings_path,measurements)
        if kind=='eval':
            e=json.loads(completed.read_text());resources=e['resources']
            count=sum(r.get('new_generated_questions',0) for r in resources)
            seconds=max((r.get('new_generation_seconds',0) for r in resources),default=0)
            if count and seconds:
                task='dev' if name.endswith('eval_dev') else e['task']
                calibration.setdefault('evaluation_questions_per_second',{})[task]=count/seconds
        else:
            t=json.loads(completed.read_text())
            calibration['last_full_training_active_seconds']=t['active_seconds']
        write_summary(a.asset_root,a.run_root,a.config)
        print('DONE',name,flush=True)

    try:
        status(phase);write_summary(a.asset_root,a.run_root,a.config)
        for method in selection['selected_methods']:
            for seed in c['training_seeds']:
                d=a.run_root/'stage2'/method/f'seed{seed}';train=d/'train'
                execute(f'stage2/{method}/seed{seed}/train','vlm_a100.train',
                    ['--output-dir',str(train),'--method',method,'--seed',str(seed),'--split','train_all',
                     '--runtime-world-size','2','--runtime-microbatch','8','--runtime-accumulation','1'],
                    train/'TRAIN_RESULT.json','train',train)
                for task in ['dev','textvqa','docvqa','mmstar']:
                    legacy=d/f'eval_{task}'
                    if (legacy/'EVALUATION.json').exists():print('reuse completed',str(legacy),flush=True);continue
                    directory=d/f'eval_{task}_a100';extra=['--output-dir',str(directory),
                        '--adapter-dir',str(train/'adapter'),'--task','textvqa' if task=='dev' else task,
                        '--split','dev' if task=='dev' else 'final','--batch-size','8']
                    if (legacy/'EVAL_INPUT_LOCK.json').exists():extra+=['--reuse-dir',str(legacy)]
                    execute(f'stage2/{method}/seed{seed}/eval_{task}','vlm_a100.evaluate',extra,
                            directory/'EVALUATION.json','eval',directory)
        phase='verification';status(phase)
        verification=verify(a.asset_root,a.run_root,a.config)
        if not verification['passed']:raise ValueError('indexed final verification failed')
        complete={'phase':'complete','status':'complete','updated_at':timestamp(),'runtime':'vlm_a100',
                  'completed_training_runs':15,'completed_evaluations':67,'config_sha256':content_sha(c)}
        write_json(a.run_root/'STATUS.json',complete)
        write_summary(a.asset_root,a.run_root,a.config,a.public_output)
        write_json(a.asset_root/'EXECUTION_STATUS.json',complete)
        print('ALL DEEP EXPERIMENTS COMPLETE WITH RECORDED A100 MIGRATION',flush=True)
    except Exception as error:
        failed={'phase':phase,'status':'failed','error':str(error),'updated_at':timestamp(),'runtime':'vlm_a100'}
        write_json(a.run_root/'STATUS.json',failed);write_json(a.asset_root/'EXECUTION_STATUS.json',failed)
        raise


if __name__=='__main__':main()
