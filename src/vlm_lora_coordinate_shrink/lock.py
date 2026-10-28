"""Direct current-study and accepted-producer locks; no recursive tensor audits."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT,SOURCE_ROOT
from vlm_probe.evaluate import sha_file,content_sha,read_json,package_versions,private_path,lock_file
from . import CANDIDATE,CONTROL

INITIAL='6a44359406c53261b254c3a19393c63187b34af09baec539d6b235ca148c0fa7'
FROZEN='47486b9a45c127a55223b496f9847e45b2a181dd225697d7d98b274a67ade929'
PRODUCER_FINAL='d86e0ce25d30c57377e660a9c35cecdc03b3db2205338da81292a9239b13c485'
COPY_PROVENANCE={'file':'src/vlm_generated_prefix_kl/lock.py',
    'sha256':'27b11e5e532725c1aa5540f68d3359e2b4bd1045ddcbd6d5af2e1dd0124e745d',
    'scope':'explicit direct-lock adaptation, no old globals patched'}


def verify_files(mapping):
    for name,digest in mapping.items():
        file=private_path(name)
        if not file.is_file() or sha_file(file)!=digest:
            raise ValueError(f'direct study file bytes differ: {file}')


def source_hashes():
    from vlm_object_eval.evaluate import source_hashes as generation_sources
    from .train import source_hashes as training_sources
    result=generation_sources();result.update(training_sources())
    for file in Path(__file__).parent.glob('*.py'):
        result[str(file.relative_to(SOURCE_ROOT))]=sha_file(file)
    for name in ('research_runtime/queue.py','vlm_joint/pipeline.py','vlm_precision/pipeline.py'):
        result[name]=sha_file(SOURCE_ROOT/name)
    for directory in ('vlm_object_behavior_replay','vlm_object_kl','vlm_generated_prefix_kl'):
        for file in (SOURCE_ROOT/directory).glob('*.py'):
            result[str(file.relative_to(SOURCE_ROOT))]=sha_file(file)
    file=PROJECT_ROOT/'tools/run_lora_coordinate_shrink_study.py'
    if file.exists():result[str(file.relative_to(PROJECT_ROOT))]=sha_file(file)
    return dict(sorted(result.items()))


def study_sources():
    return {str(PROJECT_ROOT/name if name.startswith('tools/') else SOURCE_ROOT/name):digest
            for name,digest in source_hashes().items()}


def _producer_map(config):
    root=private_path(config['producer_reference_root'])
    cfg=private_path(config['producer_reference_config'])
    required={str(cfg),*[str(root/name) for name in ('STUDY_INPUT_LOCK.json','COMPLETION.json',
        'analysis/pilot_seed42/screen/ANALYSIS.json','independent_review/PILOT_PREDICTION_REVIEW.json',
        'independent_review/TRAINING_COMPLETION_REVIEW.json','independent_review/ROOT_PILOT_ACCEPTANCE.json',
        'runs/teacher_kl32/seed42/train/TRAIN_RESULT.json')]}
    mapping=config.get('producer_reference_files_sha256')
    if not isinstance(mapping,dict) or set(mapping)!=required:
        raise ValueError('all8 exact stage32 accepted producer metadata proofs required')
    return root,cfg,dict(mapping)


def _large_parent_teacher(config):
    return str(private_path(config['parent_output_root'])/'teacher_replies/seed42/MANIFEST.json')


def _verify_parent_metadata(config,mapping):
    # The191MB immutable teacher digest is an accepted reference, not a fresh
    # parsing/hash task. Its exact14-map is bound by the accepted stage32 STUDY.
    root,_,producer_map=_producer_map(config)
    study_file=root/'STUDY_INPUT_LOCK.json'
    if sha_file(study_file)!=producer_map[str(study_file)]:
        raise ValueError('accepted producer study metadata changed')
    study=read_json(study_file)
    if study.get('parent_proof_files_sha256')!=mapping:
        raise ValueError('parent14 proof map differs from admitted producer study')
    large=_large_parent_teacher(config)
    if large not in mapping:
        raise ValueError('accepted teacher digest missing from parent proof map')
    verify_files({name:digest for name,digest in mapping.items() if name!=large})


def verify_parent_proof(config):
    mapping=config.get('parent_proof_files_sha256')
    if not isinstance(mapping,dict) or len(mapping)!=14:
        raise ValueError('all14 accepted stage31 input/teacher proof references required')
    _verify_parent_metadata(config,mapping)
    root=private_path(config['parent_output_root'])
    parent_study=read_json(root/'STUDY_INPUT_LOCK.json')
    if len(parent_study.get('source_sha256',{}))!=89:
        raise ValueError('accepted stage31 scientific source count differs')
    cfg=private_path(config['parent_config_file']);parent_config=read_json(cfg)
    for file in [cfg,*[private_path(parent_config[key]) for key in
            ('study_protocol','dataset_protocol','external_dataset_protocol','reference_config')]]:
        expected=parent_study['config_file_sha256'] if file==cfg else parent_study['files_sha256'].get(str(file))
        if expected is None or sha_file(file)!=expected:
            raise ValueError('accepted stage31 small config/protocol binding differs')
    analysis=read_json(root/'analysis/pilot_seed42/screen/ANALYSIS.json')
    accepted=read_json(root/'independent_review/ROOT_PILOT_ACCEPTANCE_RECOVERY_V1.json')
    if (analysis.get('status')!='complete' or analysis.get('candidate')!='teacher_ce32'
            or analysis.get('seeds')!=[42] or analysis.get('gate',{}).get('passed') is not False
            or analysis.get('eligible_candidates')!=[] or accepted.get('status')!='accepted_complete_negative_pilot_only'
            or accepted.get('goal_complete') is not False or accepted.get('independent_score_mismatches')!=0):
        raise ValueError('stage31 must remain an accepted negative input lineage')
    for file,digest in accepted['files_sha256'].items():
        if file in mapping and mapping[file]!=digest:raise ValueError('stage31 acceptance proof mismatch')
    return dict(mapping)


def verify_producer_reference(config,*,check_weights=False):
    """Pure metadata validation of seed42's accepted181-step unshrunk producer.

    check_weights is explicit prepare/export-only; the default never opens an
    adapter/base/checkpoint/probability tensor or the large historical teacher.
    """
    root,cfg,mapping=_producer_map(config);verify_files(mapping)
    study=read_json(root/'STUDY_INPUT_LOCK.json');producer_config=read_json(cfg)
    accepted=read_json(root/'independent_review/ROOT_PILOT_ACCEPTANCE.json')
    analysis=read_json(root/'analysis/pilot_seed42/screen/ANALYSIS.json')
    scoring=read_json(root/'independent_review/PILOT_PREDICTION_REVIEW.json')
    trained=read_json(root/'independent_review/TRAINING_COMPLETION_REVIEW.json')
    completion=read_json(root/'COMPLETION.json')
    result_file=root/'runs/teacher_kl32/seed42/train/TRAIN_RESULT.json';result=read_json(result_file)
    if (study.get('schema_version')!='rarevlm.generated_prefix_kl.study_input.v1'
            or study.get('config_path')!=str(cfg) or study.get('config_file_sha256')!=mapping[str(cfg)]
            or study.get('config_content_sha256')!=content_sha(producer_config)
            or study.get('parent_proof_files_sha256')!=config['parent_proof_files_sha256']
            or len(study.get('source_sha256',{}))!=84):
        raise ValueError('accepted stage32 producer study/config/source identity differs')
    if (accepted.get('status')!='accepted_complete_negative_pilot_only' or accepted.get('goal_complete') is not False
            or accepted.get('independent_score_mismatches')!=0
            or completion.get('status')!='completed_negative' or completion.get('goal_complete') is not False
            or analysis.get('status')!='complete' or analysis.get('candidate')!='teacher_kl32'
            or analysis.get('seeds')!=[42] or analysis.get('gate',{}).get('passed') is not False
            or analysis.get('eligible_candidates')!=[]):
        raise ValueError('producer remains accepted negative; it cannot admit new replication')
    for file,digest in accepted['files_sha256'].items():
        if file in mapping and mapping[file]!=digest:raise ValueError('producer root acceptance/proof digest differs')
    if (scoring.get('status')!='passed_independent_prediction_review' or scoring.get('new_score_mismatches')!=0
            or scoring.get('independent_new_questions_rescored')!=1959
            or scoring.get('official_analysis_sha256')!=mapping[str(root/'analysis/pilot_seed42/screen/ANALYSIS.json')]
            or scoring.get('study_input_lock_sha256')!=mapping[str(root/'STUDY_INPUT_LOCK.json')]
            or trained.get('status')!='passed_training_engineering_only' or trained.get('formal_updates')!=181
            or trained.get('actual_Adam_step_values')!=[181] or trained.get('master_tensors')!=144
            or trained.get('master_elements')!=1843200 or trained.get('study_input_lock_sha256')!=mapping[str(root/'STUDY_INPUT_LOCK.json')]):
        raise ValueError('accepted complete producer training/scoring review chain differs')
    fixed={'schema_version':'rarevlm.generated_prefix_kl.train.v1','status':'completed','condition':'teacher_kl32','seed':42,
        'completed_updates':181,'actual_object_exposures':2888,'actual_replay_exposures':5792,'actual_questions_seen':8680,
        'trainable_parameters':1843200,'initial_adapter_tensor_sha256':INITIAL,'final_lora_tensor_sha256':PRODUCER_FINAL,
        'frozen_parameter_sha256_before':FROZEN,'frozen_parameter_sha256_after':FROZEN,
        'frozen_parameters_unchanged':True,'reference_parameters_unchanged':True,
        'formal_reference_forward_questions':5792,'replay_hard_ce_backward':False,'replay_objective':'forward_kl_full_vocabulary'}
    for key,value in fixed.items():
        if result.get(key)!=value or isinstance(result.get(key),bool)!=isinstance(value,bool):
            raise ValueError('accepted producer Final invariant differs: '+key)
    if (trained.get('final_master_sha256')!=PRODUCER_FINAL or trained.get('initial_canonical_master_sha256')!=INITIAL
            or trained.get('saved_frozen_base_vision_before_after_sha256')!=FROZEN
            or trained.get('evidence_immutable_files_sha256',{}).get(str(result_file))!=mapping[str(result_file)]
            or config['coordinate_shrink']['seed42_producer_master_sha256']!=PRODUCER_FINAL
            or config['coordinate_shrink']['initial_master_sha256']!=INITIAL
            or private_path(config['initial_adapter'])!=private_path(result['initial_adapter_dir'])):
        raise ValueError('accepted producer Final/canonicalinitial independent signature differs')
    adapter=root/'runs/teacher_kl32/seed42/train/adapter'
    if check_weights:
        verify_files({str(adapter/name):digest for name,digest in result['adapter_files_sha256'].items()})
        initial=private_path(config['initial_adapter'])
        verify_files({str(initial/name):digest for name,digest in result['initial_adapter_files_sha256'].items()})
    return {'schema_version':'rarevlm.lora_coordinate_shrink.producer_reference.v1','seed':42,
        'candidate':CANDIDATE,'producer_condition':'teacher_kl32','artifact_role':'unshrunk_producer',
        'producer_result_file':str(result_file),'producer_result_file_sha256':mapping[str(result_file)],
        'producer_adapter_dir':str(adapter),'producer_adapter_files_sha256':dict(result['adapter_files_sha256']),
        'expected_final_master_sha256':PRODUCER_FINAL,'initial_canonical_master_sha256':INITIAL,
        'initial_adapter_dir':result['initial_adapter_dir'],'initial_adapter_files_sha256':dict(result['initial_adapter_files_sha256']),
        'source_binding_files_sha256':mapping,'producer_study_input_lock_sha256':mapping[str(root/'STUDY_INPUT_LOCK.json')],
        'producer_lineage_sha256':result['lineage_sha256'],'new_study_config_content_sha256':content_sha(config),
        'historical_training_updates':181,'new_training_optimizer_updates':0,'new_teacher_generation_questions':0,
        'producer_was_shrunk':False,'accepted_producer_result_is_negative':True}


def prepare_historical_producer_reference(config,*,check_weights=False):
    value=verify_producer_reference(config,check_weights=check_weights)
    file=private_path(config['output_root'])/'producer_runs'/CANDIDATE/'seed42/HISTORICAL_PRODUCER_REFERENCE.json'
    lock_file(file,value)
    return value


def verify_study(file,config_file,*,check_weights=False):
    file,config_file=private_path(file),private_path(config_file);lock=read_json(file);config=read_json(config_file)
    if (lock.get('schema_version')!='rarevlm.lora_coordinate_shrink.study_input.v1'
            or lock.get('config_file_sha256')!=sha_file(config_file) or lock.get('source_sha256')!=study_sources()
            or lock.get('packages')!=package_versions() or lock.get('candidate')!=CANDIDATE
            or lock.get('control_reference_only') is not True):
        raise ValueError('new shrink source/config/runtime identity differs')
    verify_files(lock['source_sha256'])
    large=_large_parent_teacher(config)
    for name,digest in lock['metadata_files_sha256'].items():
        if name==large:
            if config['parent_proof_files_sha256'].get(name)!=digest:raise ValueError('accepted large-teacher reference digest changed')
        else:verify_files({name:digest})
    parent_source=read_json(private_path(config['parent_output_root'])/'STUDY_INPUT_LOCK.json')['source_sha256']
    producer_source=read_json(private_path(config['producer_reference_root'])/'STUDY_INPUT_LOCK.json')['source_sha256']
    if lock.get('accepted_parent_source_sha256')!=parent_source:
        raise ValueError('explicit accepted stage31 source map differs')
    verify_files(lock['accepted_parent_source_sha256'])
    if 'accepted_producer_source_sha256' not in lock:
        raise ValueError('accepted stage32 source map must be explicit')
    if lock['accepted_producer_source_sha256']!=producer_source:
        raise ValueError('explicit accepted stage32 source map differs')
    verify_files(lock['accepted_producer_source_sha256'])
    verify_parent_proof(config)
    producer=verify_producer_reference(config,check_weights=False)
    if (lock.get('producer_reference_files_sha256')!=config['producer_reference_files_sha256']
            or lock.get('historical_producer_reference')!=producer):
        raise ValueError('study historicalproducer proof identity differs')
    href=private_path(config['output_root'])/'producer_runs'/CANDIDATE/'seed42/HISTORICAL_PRODUCER_REFERENCE.json'
    if not href.is_file() or read_json(href)!=producer:
        raise ValueError('explicit historicalproducer reference missing or changed')
    if check_weights:verify_files(lock['weight_files_sha256'])
    return lock
