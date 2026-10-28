"""Direct current-study locks; never recursively verify historical bank trees."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
import json
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import sha_file,content_sha,read_json,package_versions,private_path


def verify_files(mapping):
    for name,digest in mapping.items():
        file=private_path(name)
        if not file.is_file() or sha_file(file)!=digest:raise ValueError(f'direct study file bytes differ: {file}')


def source_hashes():
    from vlm_object_eval.evaluate import source_hashes as generation_sources
    from .train import source_hashes as training_sources
    result=generation_sources();result.update(training_sources())
    for file in Path(__file__).parent.glob('*.py'):result[str(file.relative_to(SOURCE_ROOT))]=sha_file(file)
    for name in ('research_runtime/queue.py','vlm_joint/pipeline.py','vlm_precision/pipeline.py'):
        result[name]=sha_file(SOURCE_ROOT/name)
    # Pure helper dependencies are actual source files, not only a historical
    # STUDY container. This does not traverse or hash old tensor banks.
    for directory in ('vlm_object_behavior_replay','vlm_object_kl'):
        for file in (SOURCE_ROOT/directory).glob('*.py'):
            result[str(file.relative_to(SOURCE_ROOT))]=sha_file(file)
    file=PROJECT_ROOT/'tools/run_gt_anchored_kl_study.py'
    if file.exists():result[str(file.relative_to(PROJECT_ROOT))]=sha_file(file)
    return dict(sorted(result.items()))


def study_sources():
    return {str(PROJECT_ROOT/name if name.startswith('tools/') else SOURCE_ROOT/name):digest for name,digest in source_hashes().items()}


def verify_parent_proof(config):
    mapping=config.get('parent_proof_files_sha256')
    if not isinstance(mapping,dict) or len(mapping)!=14:raise ValueError('all14 explicit accepted parent proof files required')
    verify_files(mapping)
    root=private_path(config['parent_output_root'])
    parent_study=read_json(root/'STUDY_INPUT_LOCK.json')
    if len(parent_study.get('source_sha256',{}))!=89:raise ValueError('accepted parent scientific-source set differs')
    verify_files(parent_study['source_sha256'])
    parent_config_file=private_path(config['parent_config_file']);parent_config=read_json(parent_config_file)
    small=[parent_config_file,*[private_path(parent_config[key]) for key in
        ('study_protocol','dataset_protocol','external_dataset_protocol','reference_config')]]
    for file in small:
        expected=(parent_study['config_file_sha256'] if file==parent_config_file else parent_study['files_sha256'].get(str(file)))
        if file==parent_config_file and parent_study.get('config_path')!=str(file):raise ValueError('accepted parent config path differs')
        if expected is None or sha_file(file)!=expected:raise ValueError('accepted parent config/protocol bytes differ')
    analysis_file=root/'analysis/pilot_seed42/screen/ANALYSIS.json'
    acceptance_file=root/'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json'
    if str(analysis_file) not in mapping or str(acceptance_file) not in mapping:raise ValueError('parent acceptance/analysis missing')
    analysis,accepted=read_json(analysis_file),read_json(acceptance_file)
    if (analysis.get('status')!='complete' or analysis.get('candidate')!='teacher_ce32' or analysis.get('seeds')!=[42]
        or analysis.get('gate',{}).get('passed') is not False or analysis.get('eligible_candidates')!=[]
        or accepted.get('status')!='accepted_complete_negative_pilot_only'
        or accepted.get('goal_complete') is not False or accepted.get('independent_score_mismatches')!=0):
        raise ValueError('parent must remain accepted completed negative, never a new candidate')
    for file,digest in accepted['files_sha256'].items():
        if file in mapping and mapping[file]!=digest:raise ValueError('parent acceptance/proof byte chains disagree')
    previous=root.parent/'vlm_visual_merger_kl_v1'
    completed=read_json(previous/'recovery_v1/COMPLETION.json')
    accepted34=read_json(previous/'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json')
    if completed.get('status')!='completed_negative' or completed.get('gate',{}).get('passed') is not False or accepted34.get('status')!='accepted_complete_negative_pilot_only':
        raise ValueError('current visual pilot must have complete independently accepted negative result')
    return dict(mapping)


def verify_study(file,config_file,*,check_weights=False):
    file,config_file=private_path(file),private_path(config_file);lock=read_json(file)
    if (lock.get('schema_version')!='rarevlm.gt_anchored_kl.study_input.v1'
        or lock.get('config_file_sha256')!=sha_file(config_file)
        or lock.get('source_sha256')!=study_sources() or lock.get('packages')!=package_versions()
        or lock.get('candidate')!='gt_ce_kl32' or lock.get('control_reference_only') is not False):
        raise ValueError('new generated-prefix KL source/config/runtime identity differs')
    verify_files(lock['source_sha256']);verify_files(lock['metadata_files_sha256'])
    verify_files(lock['accepted_parent_source_sha256'])
    # Weight bytes are checked once at prepare and before each GPU model load,
    # not recursively for every historical row/cell or old checkpoint/bank.
    if check_weights:verify_files(lock['weight_files_sha256'])
    return lock
