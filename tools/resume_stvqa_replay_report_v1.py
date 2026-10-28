"""Finish the frozen STVQA report after its import-only failure; no GPU work."""
import ast
from datetime import datetime, timezone
from pathlib import Path

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import content_sha, sha_file, write_json
from vlm_grpo_iteration_v1.common import read


def main():
    root = PROJECT_ROOT.parents[1] / 'outputs/vlm_stvqa_replay_v1'
    recovery = root / 'report_recovery_v1'
    if (root / 'COMPLETION.json').exists():
        raise FileExistsError('Report already completed; do not repeat recovery')
    for queue in ('training_queue', 'evaluation_queue'):
        state = read(root / queue / 'STATUS.json')
        if state['status'] != 'completed' or any(
            job['state'] != 'completed' or job['exit_code'] != 0
            for job in state['jobs'].values()
        ):
            raise ValueError('All registered training/evaluation workers must be complete')
    source = PROJECT_ROOT / 'tools/run_stvqa_replay_v1.py'
    paired_source = PROJECT_ROOT / 'src/vlm_grpo_iteration_v1/study.py'
    lock = root / 'STUDY_INPUT_LOCK.json'
    if sha_file(source) != read(lock)['files'][str(source)]:
        raise ValueError('Frozen controller source changed')
    tree = ast.parse(source.read_text(), filename=str(source))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == 'run_native_stable_kl_v1'
               and [(item.name, item.asname) for item in node.names] == [('paired', None)]]
    if len(imports) != 1:
        raise ValueError('Expected exactly the recorded import failure')
    imports[0].module = 'vlm_grpo_iteration_v1.study'
    protected = {source, paired_source, lock, root / 'controller.log',
                 root / 'TRANSFER_ANALYSIS.json', PROJECT_ROOT / 'tools/run_visual_preference_v1.py'}
    protected.update(root.glob('baseline/*/PREDICTIONS.json'))
    protected.update(root.glob('runs/*/*/*/PREDICTIONS.json'))
    protected.update(root.glob('runs/*/train/adapter/*'))
    reuse = read(root / 'CONTROL_REUSE.json')['cohorts']
    for cohort in reuse.values():
        protected.update(Path(cohort[key]) for key in ('start', 'old_ce', 'old_ce_dpo'))
    before = {str(path): sha_file(path) for path in sorted(protected) if path.is_file()}
    plan = {'status': 'frozen', 'recorded_at': datetime.now(timezone.utc).isoformat(),
            'repair': 'One ImportFrom.module in memory only: run_native_stable_kl_v1 -> vlm_grpo_iteration_v1.study',
            'correct_function': 'The identical paired implementation used by the frozen parent analyzer',
            'frozen_source_sha256': sha_file(source), 'paired_source_sha256': sha_file(paired_source),
            'transformed_ast_sha256': content_sha(ast.dump(tree)), 'protected': before,
            'new_updates': 0, 'new_generations': 0, 'scoring_or_threshold_changes': False}
    write_json(recovery / 'PLAN.json', plan)
    namespace = {'__name__': 'stvqa_report_recovery_v1', '__file__': str(source)}
    exec(compile(tree, str(source), 'exec'), namespace)
    namespace['analyze']()
    after = {path: sha_file(path) for path in before}
    if before != after:
        raise ValueError('Protected source, weights, predictions or parent analysis changed')
    write_json(recovery / 'COMPLETION.json', {'status': 'completed', 'protected_files_unchanged': True,
               'protected_count': len(before), 'new_updates': 0, 'new_generations': 0,
               'analysis_sha256': sha_file(root / 'ANALYSIS.json'),
               'decision_sha256': sha_file(root / 'DECISION.json'),
               'original_failure_log_preserved': True})
    print('STVQA_REPORT_RECOVERY_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
