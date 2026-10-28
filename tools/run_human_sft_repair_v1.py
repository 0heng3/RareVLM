"""Continue from full initial baselines into the preregistered matched study."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import write_json


def main():
    config_path = PROJECT_ROOT/'configs/vlm_human_sft_repair_v1.json'
    config = json.loads(config_path.read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'WORKFLOW.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            complete = {task: (root/'initial'/task/'COMPLETE.json').exists() for task in ('human', 'text')}
            if all(complete.values()):
                break
            status_path = root/'baseline_queue/STATUS.json'
            if status_path.exists() and json.loads(status_path.read_text())['status'] == 'failed':
                write_json(root/'WORKFLOW_PROGRESS.json', {'status': 'failed', 'phase': 'initial_baselines'})
                raise RuntimeError('Initial baseline failed; training not started')
            write_json(root/'WORKFLOW_PROGRESS.json', {'status': 'waiting_initial_baselines', 'complete': complete})
            time.sleep(15)
        write_json(root/'WORKFLOW_PROGRESS.json', {'status': 'running', 'phase': 'matched_full_training_and_eval'})
        process = subprocess.run([sys.executable, '-u', '-m', 'vlm_human_sft.study', '--config', str(config_path)], cwd=PROJECT_ROOT)
        result_path = root/'COMPLETION.json'
        if process.returncode != 0 or not result_path.exists():
            write_json(root/'WORKFLOW_PROGRESS.json', {'status': 'failed', 'phase': 'matched_study', 'exit_code': process.returncode})
            raise RuntimeError('Matched study failed; no other task was killed')
        result = json.loads(result_path.read_text())
        write_json(root/'WORKFLOW_PROGRESS.json', {'status': 'completed', 'pilot_passed': result['pilot_passed'],
                                                  'goal_complete': False})


if __name__ == '__main__':
    main()
