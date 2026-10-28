"""Direct visual-merger study proofs; no historical bank/state traversal."""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_probe.evaluate import sha_file, read_json, package_versions, private_path

STUDY_SCHEMA = 'rarevlm.visual_merger_kl.study_input.v1'
PRECEDING_ROOT_ACCEPT_SHA256 = 'f569de03cc31e41606463fc626610253fe904093f97db391f6adc4a811e24ffd'


def verify_files(mapping):
    for name, digest in mapping.items():
        file = private_path(name)
        if not file.is_file() or sha_file(file) != digest:
            raise ValueError(f'direct study file bytes differ: {file}')


def canonical_sources(runtime_sources):
    """Frozen convention: tools/ from PROJECT_ROOT; all others from SOURCE_ROOT."""
    if not isinstance(runtime_sources, dict):
        raise ValueError('runtime scientific sources must be a map')
    result = {}
    for name, digest in runtime_sources.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute() or '\\' in name
                or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('scientific source relative path escape/alias')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('scientific source SHA256 differs')
        base = PROJECT_ROOT if name.startswith('tools/') else SOURCE_ROOT
        file = (base / name).resolve()
        if not file.is_relative_to(base.resolve()) or str(file) in result:
            raise ValueError('scientific source path escape or duplicate canonical key')
        result[str(file)] = digest
    return result


def verify_runtime_sources(runtime_sources, study_source_map):
    canonical = canonical_sources(runtime_sources)
    if canonical != study_source_map:
        raise ValueError('whole runtime/study scientific source graph differs')
    return canonical


def source_hashes():
    from vlm_object_eval.evaluate import source_hashes as generation_sources
    from .train import source_hashes as training_sources
    result = generation_sources()
    for name, digest in training_sources().items():
        if name in result and result[name] != digest:
            raise ValueError('conflicting direct mathematical source digest')
        result[name] = digest
    for file in Path(__file__).parent.glob('*.py'):
        result[str(file.relative_to(SOURCE_ROOT))] = sha_file(file)
    for name in ('research_runtime/queue.py', 'vlm_joint/pipeline.py', 'vlm_precision/pipeline.py'):
        result[name] = sha_file(SOURCE_ROOT / name)
    # These are actually imported pure parent tensor/loss helpers, not the old
    # study's entire historical source/weight/bank tree.
    for directory in ('vlm_object_behavior_replay', 'vlm_object_kl'):
        for file in (SOURCE_ROOT / directory).glob('*.py'):
            result[str(file.relative_to(SOURCE_ROOT))] = sha_file(file)
    file = PROJECT_ROOT / 'tools/run_visual_merger_kl_study.py'
    if file.exists():
        result[str(file.relative_to(PROJECT_ROOT))] = sha_file(file)
    result = dict(sorted(result.items()))
    canonical_sources(result)
    return result


def study_sources():
    return canonical_sources(source_hashes())


def verify_parent_proof(config):
    from .data import validate_parent_proofs
    mapping = validate_parent_proofs(config)
    root = private_path(config['parent_output_root'])
    analysis = read_json(root / 'analysis/pilot_seed42/screen/ANALYSIS.json')
    accepted = read_json(root / 'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json')
    if (analysis.get('status') != 'complete' or analysis.get('candidate') != 'teacher_ce32'
            or analysis.get('seeds') != [42] or analysis.get('gate', {}).get('passed') is not False
            or analysis.get('eligible_candidates') != []
            or accepted.get('status') != 'accepted_complete_negative_pilot_only'
            or accepted.get('independent_score_mismatches') != 0 or accepted.get('goal_complete') is not False):
        raise ValueError('accepted R31 negative/input proof differs')
    return mapping


def verify_prior_research(config):
    """Eight N32 language proofs + six N33 negative proofs, direct metadata only."""
    language = private_path(config['language_reference_root'])
    preceding = private_path(config['preceding_negative_root'])
    language_paths = [private_path(config['language_reference_config'])] + [language / name for name in (
        'STUDY_INPUT_LOCK.json', 'COMPLETION.json', 'analysis/pilot_seed42/screen/ANALYSIS.json',
        'independent_review/PILOT_PREDICTION_REVIEW.json', 'independent_review/TRAINING_COMPLETION_REVIEW.json',
        'independent_review/ROOT_PILOT_ACCEPTANCE.json', 'runs/teacher_kl32/seed42/train/TRAIN_RESULT.json')]
    preceding_paths = [preceding / name for name in ('STUDY_INPUT_LOCK.json', 'recovery_v1/RECOVERY_INPUT_LOCK.json',
        'recovery_v1/COMPLETION.json', 'analysis/pilot_seed42/screen/ANALYSIS.json',
        'independent_review/recovery_v1/PILOT_PREDICTION_REVIEW.json', 'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json')]
    mapping = config.get('prior_research_files_sha256')
    required = {str(p) for p in language_paths + preceding_paths}
    if not isinstance(mapping, dict) or len(mapping) != 14 or set(mapping) != required:
        raise ValueError('exact8 language and6 preceding direct proof metadata required')
    preceding_accept = preceding / 'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json'
    if mapping[str(preceding_accept)] != PRECEDING_ROOT_ACCEPT_SHA256:
        raise ValueError('accepted stage33 full negative root proof differs')
    verify_files(mapping)
    for root, candidate, completion_name, acceptance_name, review_name in (
        (language, 'teacher_kl32', 'COMPLETION.json', 'independent_review/ROOT_PILOT_ACCEPTANCE.json', 'independent_review/PILOT_PREDICTION_REVIEW.json'),
        (preceding, 'lora_half32', 'recovery_v1/COMPLETION.json', 'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json', 'independent_review/recovery_v1/PILOT_PREDICTION_REVIEW.json')):
        analysis_file = root / 'analysis/pilot_seed42/screen/ANALYSIS.json'
        analysis, completion = read_json(analysis_file), read_json(root / completion_name)
        acceptance, review = read_json(root / acceptance_name), read_json(root / review_name)
        if (analysis.get('status') != 'complete' or analysis.get('candidate') != candidate
                or analysis.get('seeds') != [42] or analysis.get('gate', {}).get('passed') is not False
                or analysis.get('eligible_candidates') != [] or completion.get('status') != 'completed_negative'
                or completion.get('goal_complete') is not False
                or acceptance.get('status') != 'accepted_complete_negative_pilot_only'
                or acceptance.get('independent_score_mismatches') != 0 or acceptance.get('goal_complete') is not False
                or review.get('status') != 'passed_independent_prediction_review'
                or review.get('official_analysis_sha256') != mapping[str(analysis_file)]):
            raise ValueError('prior completed negative/control proof or source binding differs')
    training = read_json(language / 'independent_review/TRAINING_COMPLETION_REVIEW.json')
    from .initialization import INITIAL_LANGUAGE_SHA256
    if (training.get('status') != 'passed_training_engineering_only'
            or training.get('initial_canonical_master_sha256') != INITIAL_LANGUAGE_SHA256
            or training.get('formal_updates') != 181):
        raise ValueError('accepted language producer numerical proof differs')
    return {'language_files_sha256': {str(p): mapping[str(p)] for p in language_paths},
        'preceding_negative_files_sha256': {str(p): mapping[str(p)] for p in preceding_paths},
        'verification_scope': '14 direct metadata raw-byte proofs; no recursive historical source/bank/state/weight verification'}


def accepted_source_maps(config):
    """Saved declarations, already bound by direct proofs; do not rehash their tree."""
    parent = read_json(private_path(config['parent_output_root']) / 'STUDY_INPUT_LOCK.json')
    language = read_json(private_path(config['language_reference_root']) / 'STUDY_INPUT_LOCK.json')
    preceding = read_json(private_path(config['preceding_negative_root']) / 'STUDY_INPUT_LOCK.json')
    recovery = read_json(private_path(config['preceding_negative_root']) / 'recovery_v1/RECOVERY_INPUT_LOCK.json')
    result = {'accepted_parent_source_sha256': parent['source_sha256'],
        'accepted_language_source_sha256': language['source_sha256'],
        'accepted_preceding_source_sha256': preceding['source_sha256'],
        'accepted_preceding_recovery_source_sha256': recovery['recovery_source_sha256']}
    if [len(v) for v in result.values()] != [89, 84, 96, 11]:
        raise ValueError('accepted prior source declaration graph sizes differ')
    return result


def verify_study(file, config_file, *, check_weights=False):
    file, config_file = private_path(file), private_path(config_file)
    lock, config = read_json(file), read_json(config_file)
    if (lock.get('schema_version') != STUDY_SCHEMA or lock.get('config_path') != str(config_file)
            or lock.get('config_file_sha256') != sha_file(config_file)
            or lock.get('packages') != package_versions() or lock.get('candidate') != 'merger_kl32'
            or lock.get('control_reference_only') is not True):
        raise ValueError('new visual-merger study source/config/runtime identity differs')
    verify_runtime_sources(source_hashes(), lock.get('source_sha256'))
    verify_files(lock['source_sha256'])
    verify_parent_proof(config)
    verify_prior_research(config)
    if lock.get('parent_proof_files_sha256') != config['parent_proof_files_sha256'] or lock.get('prior_research_files_sha256') != config['prior_research_files_sha256']:
        raise ValueError('direct parent/prior proof declarations differ')
    if any(lock.get(key) != value for key, value in accepted_source_maps(config).items()):
        raise ValueError('accepted source declaration domains differ')
    historical_bank = config['seed42_input_reference']['teacher_reply_manifest_file']
    if historical_bank in lock['metadata_files_sha256']:
        raise ValueError('historical large teacher bank must remain a proof declaration, not a recursive read')
    verify_files(lock['metadata_files_sha256'])
    from .initialization import common_initializer_manifest, initialization_paths
    initialization = common_initializer_manifest(config, verify_file_bytes=False)
    if (lock.get('common_initialization_identity') != initialization['identity']
            or lock['metadata_files_sha256'].get(str(initialization_paths(config)['manifest'])) != sha_file(initialization_paths(config)['manifest'])):
        raise ValueError('shared visual initialization stable identity/manifest differs')
    if check_weights:
        verify_files(lock['weight_files_sha256'])
        common_initializer_manifest(config, verify_file_bytes=True)
    return lock
