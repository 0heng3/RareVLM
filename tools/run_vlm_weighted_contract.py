"""Own-resource real-QA contract checks before the weighted-SFT research queue."""
from __future__ import annotations
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vlm_joint.pipeline import gpu_snapshot, qualified
from vlm_precision.pipeline import GPU_LOCK_ROOT, now, read_json, sha_file, write_json
from vlm_weighted.data import load_weights, read_weight_lock
from vlm_weighted.train import assert_scientific_profile

ROOT = Path('/opt/data/private/longtail_vlm/rarevlm_pilot/outputs/vlm_weighted_sft_v1')
ASSET = ROOT.parent/'vlm_deep_v2'
MODEL = ROOT.parent/'vlm_sft_textvqa_v1/model'
CONFIG = REPO/'configs/vlm_weighted_sft_v1.json'
STATUS = ROOT/'contracts/GPU_CONTRACT_STATUS.json'


def main():
    config = read_json(CONFIG)
    assert_scientific_profile(config, 'qv_r8', 42, 'train_all')
    weight_lock = read_weight_lock(ROOT/'WEIGHT_LOCK.json', ASSET/'prepared/manifest.json')
    root = ROOT/'contracts'
    root.mkdir(exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    owner = (root/'GPU_CONTRACT.lock').open('a+')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    streams = []
    record = {'schema_version':'rarevlm.weighted_gpu_contract.v1', 'pid':os.getpid(),
        'started_at':now(), 'scope':'nonresearch real33QA/3updates; resume exercise',
        'config_file_sha256':sha_file(CONFIG), 'weight_lock_file_sha256':sha_file(ROOT/'WEIGHT_LOCK.json'),
        'source_sha256':{**{str(f.relative_to(REPO / 'src')):sha_file(f) for f in (REPO / "src" / 'vlm_weighted').glob('*.py')},
                         'rarevlm_paths.py':sha_file(REPO / 'src/rarevlm_paths.py')},
        'existing_tasks_untouched':True, 'jobs':[]}
    def status(phase, **extras):
        write_json(STATUS, {**record, 'phase':phase, 'updated_at':now(), **extras})
    try:
        idle = {}
        while True:
            devices = gpu_snapshot()
            for device in devices.values():
                uuid = device['uuid']
                idle[uuid] = idle.get(uuid, 0)+1 if qualified(device) else 0
            candidates = [d for d in devices.values() if idle.get(d['uuid'],0)>=3]
            status('waiting_resources', idle_confirmations=idle)
            if len(candidates)>=2:
                chosen = sorted(candidates, key=lambda d:d['index'])[:2]
                try:
                    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
                    for device in sorted(chosen, key=lambda d:d['uuid']):
                        stream = (GPU_LOCK_ROOT/(device['uuid']+'.lock')).open('a+')
                        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError: stream.close(); raise
                        streams.append(stream)
                    fresh = gpu_snapshot()
                    if all(fresh[d['index']]['uuid']==d['uuid'] and qualified(fresh[d['index']]) for d in chosen):
                        break
                except BlockingIOError:
                    pass
                for stream in streams: stream.close()
                streams.clear()
            time.sleep(10)
        record['admission_snapshot'] = chosen
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in chosen),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
            PYTHONPATH=str(REPO / 'src') + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
        common = [sys.executable,'-m','torch.distributed.run','--nnodes','1','--nproc_per_node','2',
            '--master_port','29741','-m','vlm_weighted.train','--asset-root',str(ASSET),
            '--model-root',str(MODEL),'--method','qv_r8','--seed','42','--split','train_all',
            '--weight-lock',str(ROOT/'WEIGHT_LOCK.json'),'--config',str(CONFIG),'--contract-questions','33']
        jobs = [('uniform','uniform33',None),('frequency','frequency_resume33',2),
                ('frequency','frequency_resume33',None)]
        for index,(condition,name,stop) in enumerate(jobs):
            directory = root/name
            cmd = common+['--condition',condition,'--output-dir',str(directory)]
            if stop is not None: cmd += ['--contract-stop-after-updates',str(stop)]
            item = {'index':index,'condition':condition,'output_dir':str(directory),
                'stop_after':stop,'started_at':now(),'command':cmd}
            record['jobs'].append(item)
            log_path = ROOT/'logs'/f'gpu_contract_{index}_{condition}.log'
            with log_path.open('a') as log:
                process = subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(s.fileno() for s in streams))
                item['pid'] = process.pid
                while process.poll() is None:
                    status('running', current_job=index, progress=read_json(directory/'PROGRESS.json'))
                    time.sleep(5)
                item['returncode'] = process.returncode
                item['finished_at'] = now()
                if process.returncode: raise RuntimeError(f'contract job{index} failed; see {log_path}')
        uniform = read_json(root/'uniform33/TRAIN_RESULT.json')
        weighted = read_json(root/'frequency_resume33/TRAIN_RESULT.json')
        for result in (uniform,weighted):
            if (result['completed_updates']!=3 or result['actual_questions_seen']!=33
                or not result['frozen_parameters_unchanged']):
                raise ValueError('real33QA contract failed')
        if weighted['resumed_at_update']!=2:
            raise ValueError('resume did not continue from update2')
        for key in ('initial_trainable_tensor_sha256','initialization_contract_sha256',
                    'global_question_order_sha256','frozen_parameter_sha256_before'):
            if uniform[key]!=weighted[key]: raise ValueError(f'paired initial contract differs:{key}')
        manifest = read_json(ASSET/'prepared/manifest.json')
        qids = [r['question_id'] for r in manifest['splits']['train_all'][:33]]
        expected = sum(load_weights(weight_lock,qids,condition='frequency'))
        if not math.isclose(weighted['actual_question_weight_sum'],expected,abs_tol=1e-4):
            raise ValueError('weighted QA count differs after resume')
        if not math.isclose(uniform['actual_question_weight_sum'],33,abs_tol=1e-6):
            raise ValueError('uniform objective changed')
        ready = {**record, 'complete':True,'finished_at':now(),
            'checks':['same fresh initializer','same33QA order','3updates/33realQA','sentinel final1QA',
                      'frequency pause2/resume3','full-training fixed weight sum','finite gradients',
                      'unchanged frozen parameters'],
            'uniform_result_sha256':sha_file(root/'uniform33/TRAIN_RESULT.json'),
            'frequency_result_sha256':sha_file(root/'frequency_resume33/TRAIN_RESULT.json'),
            'expected_contract_weight_sum':expected}
        write_json(ROOT/'GPU_CONTRACT_READY.json',ready)
        status('complete', checks=ready['checks'])
        print('GPU CONTRACT PASSED', now(), flush=True)
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for stream in streams: stream.close()
        owner.close()


if __name__=='__main__': main()
