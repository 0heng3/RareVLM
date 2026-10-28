"""New loss/adapter identity and explicit accepted historical JSON references."""
COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_generated_prefix_kl/contracts.py', 'sha256': 'bef2f992a41f957ddacce31cda027137a50484f175ade71fb09c3f83d6f83aee', 'scope': 'independent visual-merger copy; original modules/globals are untouched'}
VISUAL_DIRECT_COPY_PROVENANCE={'source': 'src/vlm_lora_coordinate_shrink/contracts.py', 'sha256': 'd4be2719cb741959fc0b9cbb78d056da15f6d6fc82ebc05e30f57685caaa319d', 'scope': 'explicit independent visual-merger copy; frozen originals and globals unchanged'}
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
import json
import math
from pathlib import Path
from vlm_object_eval import evaluate as generation
from vlm_posttrain_eval import evaluate as public_text
from vlm_object_kl_eval.contracts import (FIXED,path,canonical_rows,external_data,load_data,
    external_admission as frozen_external_admission,validate_parameter_budget)
from vlm_probe.evaluate import (content_sha,lock_file,package_versions,processor_for,private_path,read_json,sha_file)
from . import CANDIDATE,SCHEMA,CONFIG_SCHEMA
from .lock import source_hashes,study_sources,verify_study,verify_parent_proof,verify_files
CONTROL='teacher_kl32'
TRAINED_CONDITIONS=(CANDIDATE,)
CONDITIONS=('initial',CONTROL,CANDIDATE)
TASKS=('object','text','external','pope')


def validate_evaluation_dtypes(value,*,composite):
    expected={'parameter_tensors_by_dtype':{'bfloat16':972 if composite else 968},
        'parameter_stored_elements_by_dtype':{'bfloat16':3758309376 if composite else 3756466176},
        'lora_tensors_by_dtype':{'bfloat16':148 if composite else 144},
        'lora_elements_by_dtype':{'bfloat16':3686400 if composite else 1843200},
        'buffer_tensors_by_dtype':{'float32':38},'params4bit_tensors':0,'trainable_parameters':0,
        'note':'NF4 uint8 stored elements are packed and are not an original parameter count'}
    if value!=expected or content_sha(value)!=content_sha(expected):
        raise ValueError('native BF16 inventory differs from the exact declared four-tensor increment')
    return expected


def assert_pair(before,after,task):
    """Exact inputs and metadata; only the declared extra4 BF16 inventory differs."""
    from vlm_object_kl_eval.analysis import SCIENCE_FIELDS
    from vlm_posttrain_pope.analysis import AUDIT_FIELDS
    aliases=before.get('comparison_metadata',{})
    for cell in (before,after):
        validate_evaluation_dtypes(cell['lock'].get('actual_parameter_dtypes'),
            composite=cell.get('condition')==CANDIDATE)
    for key in SCIENCE_FIELDS:
        if key=='actual_parameter_dtypes':continue
        value=aliases.get(key) if key in ('data_manifest_sha256','data_manifest_file_sha256','questions','unique_images') and aliases else before['lock'].get(key)
        if value is None or value!=after['lock'].get(key):raise ValueError('paired scientific metadata differs: '+key)
    if len(before['rows'])!=len(after['rows']):raise ValueError('paired full split lengths differ')
    for left,right in zip(before['rows'],after['rows']):
        for key in ('question_id','image_id',*AUDIT_FIELDS):
            if key not in left or key not in right or left[key]!=right[key]:raise ValueError('paired actual input differs: '+key)
        if task!='text':
            if any(left[key]!=right[key] for key in ('label','category')):raise ValueError('paired binary reference differs')
        elif 'category' in left or 'category' in right:raise ValueError('only empty Text category aliases may be canonicalized')

UPSTREAM_COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_object_behavior_replay/contracts.py', 'sha256': '859a675b3904dcf7d1a116d4eaf287ddcd20cf70093e63765403bfa92dc761c1', 'copied_functions': ['directory', 'validate_inherited_admissions', 'create_inherited_admissions', 'validate_historical_cohort_scope', 'historical_evaluation_view'], 'scope': 'independent copies; no old globals patched'}

def directory(root,condition,seed,task):
    return root/'runs'/condition/f'seed{seed}'/task

def validate_inherited_admissions(config):
    """New alias files retain the exact old qualified evidence, without re-review."""
    result={}
    definitions=[('data_admission','reference_data_admission',
        ('status','quality_gate_passed','scope','automatic_diagnostic_bypass_allowed','train_manifest_sha256','dataset_protocol_sha')),
        ('evaluation_data_admission','reference_evaluation_data_admission',
        ('schema_version','status','quality_gate_passed','scope','automatic_diagnostic_bypass_allowed',
         'external_manifest_sha256','external_class_balanced_manifest_sha256','external_dataset_protocol_sha256',
         'frozen_audit_sha256','semantic_review_sha256','cpu_completion_sha256'))]
    for name,reference,fields in definitions:
        current_file,reference_file=path(config,name),path(config,reference)
        if current_file==reference_file:raise ValueError('new study requires an explicit inherited-admission alias')
        current,original=read_json(current_file),read_json(reference_file)
        if (current.get('source_admission_path')!=str(reference_file)
            or current.get('source_admission_sha256')!=sha_file(reference_file)):
            raise ValueError('inherited admission original byte binding differs')
        for field in fields:
            if field not in original or current.get(field)!=original[field]:
                raise ValueError(f'inherited admission scientific field differs: {name}/{field}')
        if (current.get('status')!='admitted' or current.get('quality_gate_passed') is not True
            or current.get('automatic_diagnostic_bypass_allowed') is not False):
            raise ValueError('qualified factual data admission required; no diagnostic bypass')
        expected_scope='annotation_supervised_object_sft' if name=='data_admission' else 'object_external_confirmation'
        if current.get('scope')!=expected_scope:raise ValueError('inherited admission task scope differs')
        result[name+'_sha256']=sha_file(current_file);result[reference+'_sha256']=sha_file(reference_file)
    training=read_json(path(config,'data_admission'))
    if (training['train_manifest_sha256']!=sha_file(path(config,'train_manifest'))
        or training['dataset_protocol_sha']!=sha_file(path(config,'dataset_protocol'))):
        raise ValueError('inherited object train/protocol bytes differ')
    return result

def create_inherited_admissions(config):
    """Clean setup copies qualified evidence, never invents or overwrites admission."""
    root=path(config,'output_root')
    original_train=read_json(path(config,'reference_data_admission'))
    if (original_train.get('status')!='admitted' or original_train.get('quality_gate_passed') is not True
        or original_train.get('scope')!='annotation_supervised_object_sft'
        or original_train.get('automatic_diagnostic_bypass_allowed') is not False
        or original_train.get('train_manifest_sha256')!=sha_file(path(config,'train_manifest'))
        or original_train.get('dataset_protocol_sha')!=sha_file(path(config,'dataset_protocol'))):
        raise ValueError('original qualified object admission cannot be inherited')
    # The frozen verifier checks original external schema/evidence/raw byte contracts.
    frozen_external_admission({**config,'evaluation_data_admission':str(path(config,'reference_evaluation_data_admission'))})
    aliases={}
    for name,reference,filename in (('data_admission','reference_data_admission','DATA_ADMISSION.json'),
        ('evaluation_data_admission','reference_evaluation_data_admission','EVALUATION_DATA_ADMISSION.json')):
        file=path(config,name);original_file=path(config,reference)
        if file!=root/filename or file==original_file:raise ValueError('admission alias must be a new-root file')
        if (root/'STUDY_INPUT_LOCK.json').exists() and not file.exists():
            raise ValueError('frozen study admission is missing; no silent recreation')
        original=read_json(original_file);value=dict(original)
        previous={key:original[key] for key in ('source_admission_path','source_admission_sha256') if key in original}
        if previous:value['inherited_original_source_provenance']=previous
        value.update(source_admission_path=str(original_file),source_admission_sha256=sha_file(original_file))
        aliases[file]=value
    # lock_file accepts identical existing bytes/content and rejects differences without replacement.
    for file,value in aliases.items():lock_file(file,value)
    return validate_inherited_admissions(config)

def validate_historical_cohort_scope(records,refs,image_lookup,metadata,evaluation,lock,task):
    """Image lookup catalogs may span all splits; counts belong to selected refs."""
    selected={row['image_id'] for row in refs};predicted={row['image_id'] for row in records}
    if not selected or predicted!=selected or not selected.issubset(image_lookup):
        raise ValueError('historical selected image cohort differs or lacks lookup members')
    questions,images=len(refs),len(selected)
    if 'questions' not in metadata or 'unique_images' not in metadata:
        raise ValueError('historical canonical metadata lacks selected scope')
    question_aliases=('n','questions') if task=='text' else ('questions',)
    if not any(key in evaluation for key in question_aliases) or 'unique_images' not in evaluation:
        raise ValueError('historical evaluation lacks original declared scope')
    if not any(key in lock for key in question_aliases) or not any(key in lock for key in ('expected_images','unique_images')):
        raise ValueError('historical input lock lacks original declared scope')
    for name,scope in (('metadata',metadata),('evaluation',evaluation),('input_lock',lock)):
        for key in ('questions','n','expected_questions'):
            if key in scope and (type(scope[key]) is not int or scope[key]!=questions):
                raise ValueError(f'historical selected question scope differs: {name}/{key}')
        for key in ('unique_images','expected_images'):
            if key in scope and (type(scope[key]) is not int or scope[key]!=images):
                raise ValueError(f'historical selected image scope differs: {name}/{key}')
    return selected

def historical_evaluation_view(original,task):
    """Only old POPE lacks a percent field; original JSON bytes stay authoritative."""
    view=dict(original);changed=[];origin='original.score_percent'
    if 'score_percent' not in original:
        if task!='pope':raise ValueError('missing historical percent is allowed only for original POPE schema')
        overall=original.get('metrics',{}).get('overall',{});accuracy=overall.get('accuracy')
        if type(accuracy) not in (int,float) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
            raise ValueError('historical POPE official accuracy fraction missing or invalid')
        if 'n' in overall and overall['n']!=original['questions']:
            raise ValueError('historical POPE official denominator differs')
        view['score_percent']=100*accuracy;changed=['score_percent'];origin='original.metrics.overall.accuracy * 100'
    value=view['score_percent']
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=100:
        raise ValueError('historical score percent is invalid')
    restored={key:value for key,value in view.items() if key not in changed}
    if restored!=original:raise ValueError('historical evaluation normalization changed original fields')
    return view,{'task':task,'changed_fields':changed,'score_percent_source':origin,'in_memory_only':True,
        'original_evaluation_content_sha256':content_sha(original),
        'all_original_fields_sha256_after':content_sha(restored),'original_fields_unchanged':True}

def external_admission(config):
    validate_inherited_admissions(config)
    return frozen_external_admission(config)


def validate_condition(condition,seed,task=None):
    if condition!=CANDIDATE or seed not in (0,42,200):
        raise ValueError('only merger_kl32 at registered order seeds may train/evaluate')
    if task is not None and task not in TASKS:raise ValueError('unknown evaluation task')


def validate_historical_request(condition,seed,task):
    if seed!=42 or condition not in ('initial',CONTROL) or task not in TASKS:
        raise ValueError('historical references use the one shared seed42 checkpoint only')
    if condition==CONTROL and task not in ('object','text'):
        raise ValueError('historical language KL is an internal reference, never selected or confirmed')


def verified_model_identity(verification):
    if not isinstance(verification,dict) or set(verification)!={'identity','telemetry'}:
        raise ValueError('model verification must explicitly separate stable identity and telemetry')
    identity=verification['identity'];telemetry=verification['telemetry']
    if not isinstance(identity,dict) or not identity or not isinstance(telemetry,dict):raise ValueError('model verification domain missing')
    def check(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key.endswith('seconds') or key in ('elapsed','wall_time','timestamp'):raise ValueError('dynamic telemetry in scientific identity')
                check(item)
        elif isinstance(value,list):
            for item in value:check(item)
    check(identity)
    for key,value in telemetry.items():
        if not key.endswith('seconds') or type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('telemetry permits finite nonnegative seconds only')
    normalized=json.loads(json.dumps(identity,ensure_ascii=False,allow_nan=False))
    if normalized!=identity:raise ValueError('model identity must be stable JSON-native metadata')
    return normalized


def composite_adapter_files(adapter_dir,model_root):
    from .model import validate_composite_adapter_config
    adapter_dir=private_path(adapter_dir)
    files={file.name:sha_file(file) for file in sorted(adapter_dir.iterdir()) if file.is_file()}
    if not {'adapter_config.json','adapter_model.safetensors'}.issubset(files):
        raise ValueError('complete composite safetensors adapter required')
    validate_composite_adapter_config(read_json(adapter_dir/'adapter_config.json'),model_root)
    return files


def adapter_binding(config,condition,seed,adapter_dir,model_root,*,config_file):
    initial=path(config,'initial_adapter')
    if condition=='initial':
        if seed!=42 or private_path(adapter_dir)!=initial:raise ValueError('one shared fixed initial checkpoint only')
        files,binding=public_text.adapter_binding(config,'initial',42,initial,model_root)
        return files,{**binding,'shared_initial_checkpoint':True}
    validate_condition(condition,seed)
    place=directory(path(config,'output_root'),condition,seed,'train')
    if private_path(adapter_dir)!=place/'adapter':raise ValueError('only complete new visual-merger Final adapter accepted')
    from .inputs import verify_training_result
    from .model import verify_training_model_metadata
    from .initialization import common_initializer_manifest
    provenance=verify_training_result(config,place,condition=condition,seed=seed)
    file=place/'TRAIN_RESULT.json';result=read_json(file);train_lock=read_json(place/'TRAIN_INPUT_LOCK.json')
    fixed={'schema_version':SCHEMA,'status':'completed','condition':condition,'seed':seed,'completed_updates':181,
        'actual_object_exposures':2888,'actual_replay_exposures':5792,'trainable_parameters':1843200,
        'replay_hard_ce_backward':False,'replay_objective':'forward_kl_full_vocabulary','reference_parameters_unchanged':True,
        'formal_reference_forward_questions':5792}
    for key,value in fixed.items():
        if result.get(key)!=value or type(result.get(key)) is not type(value):raise ValueError('completed visual-merger training differs: '+key)
    pref=place/'PREFLIGHT.json'
    if sha_file(pref)!=result.get('preflight_file_sha256') or read_json(pref)!=result.get('preflight'):
        raise ValueError('embedded preflight raw file/result content differs')
    identity=verified_model_identity(verify_training_model_metadata(result,train_lock,config))
    init=common_initializer_manifest(config,verify_file_bytes=True)
    files=composite_adapter_files(place/'adapter',model_root)
    if files!=result['adapter_files_sha256']:raise ValueError('complete composite adapter file bytes changed')
    return files,{'schema_version':'rarevlm.visual_merger_kl.adapter_binding.v1','path':str(file),'sha256':sha_file(file),
        'condition':condition,'seed':seed,'artifact_role':'frozen_language144_trainable_visual_merger4',
        'model_identity':identity,'common_initialization_identity':init['identity'],
        'training_provenance':provenance,'preflight_file_sha256':sha_file(pref)}


def validate_legacy_microbatch_digest(anchor,schedule):
    """Missing old serialization is derivable only from its locked full content."""
    if anchor.get('schedule_sha256')!=content_sha(schedule):raise ValueError('historical schedule content chain differs')
    required={'object_question_ids_by_update','replay_question_ids_by_update','learning_rates_by_update',
        'training_microbatch_questions','object_microbatches_by_update','replay_microbatches_by_update'}
    if set(schedule)!=required or schedule['training_microbatch_questions']!=4:
        raise ValueError('complete registered schedule layout required')
    rates=schedule['learning_rates_by_update']
    if len(rates)!=181 or any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in rates):
        raise ValueError('complete finite181 learning-rate plan required')
    for source,total,width in [('object',2888,16),('replay',5792,32)]:
        rows=schedule[source+'_question_ids_by_update'];micro=schedule[source+'_microbatches_by_update']
        if len(rows)!=181 or len(micro)!=181 or sum(map(len,rows))!=total:raise ValueError('complete181 exposure plan differs')
        for i,(q,b) in enumerate(zip(rows,micro)):
            if len(q)!=(8 if source=='object' and i==180 else width) or b!=[q[j:j+4] for j in range(0,len(q),4)]:
                raise ValueError('historical four-QA microbatch boundaries differ')
    digest=content_sha(schedule['replay_microbatches_by_update'])
    if 'replay_microbatch_order_sha256' in anchor and anchor['replay_microbatch_order_sha256']!=digest:
        raise ValueError('declared historical replay microbatch digest differs')
    return digest








def parse_history(raw):
    if not raw.endswith(b'\n'):raise ValueError('historical complete JSONL interrupted')
    rows=[]
    for line in raw.splitlines():
        value=json.loads(line)
        if not isinstance(value,dict) or 'question_id' not in value:raise ValueError('historical prediction item schema differs')
        rows.append(value)
    if not rows:raise ValueError('historical full prediction file empty')
    return rows


def historical_cell(root,condition,task,config_file,config):
    if condition not in ('initial',CONTROL) or (condition==CONTROL and task not in ('object','text')):
        raise ValueError('historical CE is internal reference only, never an eligible model')
    study=verify_study(root/'STUDY_INPUT_LOCK.json',config_file)
    source=study['historical_sources'][f'{condition}_42_{task}']
    for kind in ('predictions','evaluation','input_lock'):
        if sha_file(source[kind+'_file'])!=source[kind+'_sha256']:raise ValueError('accepted historical raw bytes differ: '+kind)
    evaluation=read_json(source['evaluation_file']);lock=read_json(source['input_lock_file'])
    rows=parse_history(Path(source['predictions_file']).read_bytes());refs,images,_,metadata=load_data(config,task)
    selected=validate_historical_cohort_scope(rows,refs,images,metadata,evaluation,lock,task)
    view,view_proof=historical_evaluation_view(evaluation,task)
    if (evaluation.get('status')!='completed' or evaluation['predictions_sha256']!=source['predictions_sha256']
        or evaluation['input_lock_sha256']!=content_sha(lock) or len(rows)!=len(refs)
        or len({r['question_id'] for r in rows})!=len(refs) or len({r['image_id'] for r in rows})!=len(selected)):
        raise ValueError('accepted historical complete coverage/input/raw-byte binding differs')
    for key,value in FIXED.items():
        if lock.get(key)!=value:raise ValueError('historical fixed generation contract differs: '+key)
    for row,ref in zip(rows,refs):
        if (row['question_id']!=ref['question_id'] or row['image_id']!=ref['image_id']
            or row['question_sha256']!=content_sha(ref['question'])
            or row['question_prompt_sha256']!=content_sha(ref['question']+generation.suffix(task))
            or row.get('input_truncated') is not False or row['input_lock_sha256']!=content_sha(lock)
            or not isinstance(row.get('prediction'),str) or not isinstance(row.get('generated_ids'),list)
            or content_sha(row['generated_ids'])!=row.get('generated_ids_sha256')
            or type(row.get('score')) not in (int,float) or not math.isfinite(row['score'])):
            raise ValueError('historical qid/prompt/actual-input/generated-ID identity differs')
        if task!='text' and any(row.get(key)!=ref[key] for key in ('label','category')):
            raise ValueError('historical binary scoring identity differs')
    # Reuse accepted per-item scores and official metrics. No model/processor or baseline rescoring.
    mean=100*sum(row['score'] for row in rows)/len(rows)
    if not math.isclose(mean,view['score_percent'],rel_tol=0,abs_tol=1e-10):raise ValueError('historical saved score denominator differs')
    canonical,proof=canonical_rows(rows,task)
    declaration={'schema_version':'rarevlm.visual_merger_kl.historical_reference.v1','condition':condition,'seed':42,'task':task,
        'new_generation_questions':0,'new_scoring_calls':0,'source_data':source,'parent_acceptance_chain_sha256':study['parent_proof_files_sha256'],
        'prior_research_acceptance_chain_sha256':study['prior_research_files_sha256'],
        'new_study_config_sha256':sha_file(config_file),'text_category_canonicalization':proof,
        'evaluation_view_provenance':view_proof,'scope':'accepted immutable raw historical reference; original source/hardware retained'}
    lock_file(directory(root,condition,42,task)/'HISTORICAL_REFERENCE.json',declaration)
    return {'complete':True,'condition':condition,'seed':42,'task':task,'rows':canonical,'lock':lock,
        'evaluation':view,'canonicalization':proof,'comparison_metadata':metadata,'provenance':source}


def phase_admission(file,config_file,phase):
    value=read_json(private_path(file))
    if (value.get('schema_version')!=f'rarevlm.visual_merger_kl.{phase}.v1' or value.get('status')!='complete'
        or value.get('candidate')!=CANDIDATE or value.get('seeds')!=[0,42,200]
        or value.get('config_sha256')!=sha_file(config_file) or value.get('source_sha256')!=source_hashes()
        or value.get('gate',{}).get('passed') is not True):
        raise ValueError('unchanged full three-seed prior phase must pass before confirmation')
    verify_source_data(value['source_data'])
    if phase=='external':
        phase_admission(value['screen_analysis_path'],config_file,'screen')
        if sha_file(value['screen_analysis_path'])!=value['screen_analysis_sha256']:raise ValueError('internal screen raw bytes changed')
    return value


def verify_source_data(sources):
    if not sources:raise ValueError('scored source provenance missing')
    for source in sources.values():
        for kind in ('predictions','evaluation','input_lock'):
            if sha_file(source[kind+'_file'])!=source[kind+'_sha256']:raise ValueError('scored source raw bytes changed: '+kind)


def validate_gpu_admission(config_file,admission_file,*,condition,seed,task,gpu_uuid,shard=None,
                           plan_file_sha256=None,prompt_inputs_file_sha256=None):
    config_file=private_path(config_file);config=read_json(config_file);root=path(config,'output_root');study_file=root/'STUDY_INPUT_LOCK.json'
    verify_study(study_file,config_file)
    if task=='teacher_rollout':
        if condition!='initial' or seed not in (0,200) or shard not in (0,1):raise ValueError('new teacher rollouts only eligible replicated seeds')
        role=f'teacher_seed{seed}_shard{shard}'
    else:
        validate_condition(condition,seed,task);role=f'{"train_eval" if task in ("object","text") else task}_{condition}_seed{seed}'
    expected={'schema_version':'rarevlm.visual_merger_kl.gpu_admission.v1','status':'admitted','condition':condition,'seed':seed,'task':task,
        'role':role,'gpu_uuid':gpu_uuid,'config_sha256':sha_file(config_file),'config_file_sha256':sha_file(config_file),
        'study_input_lock_path':str(study_file),'study_input_lock_sha256':sha_file(study_file)}
    if task=='teacher_rollout':
        expected.update(shard=shard,plan_file_sha256=plan_file_sha256,prompt_inputs_file_sha256=prompt_inputs_file_sha256)
        if plan_file_sha256!=sha_file(root/'prepared'/f'seed{seed}'/'PLAN.json') or prompt_inputs_file_sha256!=sha_file(root/'prepared'/f'seed{seed}'/'PROMPT_INPUTS.jsonl'):
            raise ValueError('teacher actual planned prompt inputs differ')
        from .analysis import passing_pilot
        passing_pilot(root/'analysis/pilot_seed42/screen/ANALYSIS.json',config_file)
    admission=read_json(private_path(admission_file))
    for key,value in expected.items():
        if admission.get(key)!=value:raise ValueError('new GPU admission identity differs: '+key)
    return admission


def read_cell(root, condition, seed, task, config_file, config, processor=None):
    if condition in ('initial',CONTROL):validate_historical_request(condition,seed,task)
    else:validate_condition(condition,seed,task)
    verify_study(root/'STUDY_INPUT_LOCK.json',config_file)
    place = directory(root, condition, seed, task)
    if condition in ("initial",CONTROL):
        return historical_cell(root,condition,task,config_file,config)
    result_file = place / "EVALUATION.json"
    if not result_file.exists():
        return {"complete": False, "condition": condition, "seed": seed, "task": task, "path": str(place)}
    lock, result = read_json(place / "INPUT_LOCK.json"), read_json(result_file)
    rows, images, _, metadata = load_data(config, task)
    expected = {**FIXED, **metadata, "condition": condition, "seed": seed, "task": task,
        "schema_version": "rarevlm.visual_merger_kl.eval_input.v1", "study_config_sha256": sha_file(config_file),
        "study_config_content_sha256": content_sha(config), "source_sha256": source_hashes(),
        "packages": package_versions(), "prompt_suffix": generation.suffix(task)}
    if task == "external":
        expected.update(external_admission(config))
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"completed visual-merger cell identity differs: {key}")
    files, binding = adapter_binding(config, condition, seed, Path(lock["adapter_dir"]),
        path(config, "model_root"), config_file=config_file)
    if lock.get('actual_composite_identity')!=binding['model_identity']['evaluation_identity']:
        raise ValueError('saved actual composite/native/base/language identity differs')
    validate_evaluation_dtypes(lock.get('actual_parameter_dtypes'),composite=True)
    if files != lock["adapter_files_sha256"] or binding != lock["training_binding"]:
        raise ValueError("completed visual-merger training/teacher/adapter bytes changed")
    if task in ("external", "pope") and condition != "initial":
        previous_file = Path(lock["prior_phase_analysis_file"])
        previous = phase_admission(previous_file, config_file, "screen" if task == "external" else "external")
        if sha_file(previous_file) != lock["prior_phase_analysis_sha256"]:
            raise ValueError("prior phase analysis byte binding changed")
        source_task = "text" if task == "external" else "external"
        if previous["source_data"][f"{condition}_{seed}_{source_task}"].get("training_result_sha256") != binding["sha256"]:
            raise ValueError("confirmation adapter differs from its passing prior phase")
    if processor is None:
        processor = processor_for(path(config, "model_root"), read_json(path(config, "reference_config")), 1024)
    predictions = place / "predictions.jsonl"
    records, repaired = generation.complete_cache(predictions, rows, images, lock, processor, repair=False)
    if (len(records) != len(rows) or repaired or result.get("status") != "completed"
        or result.get("schema_version") != "rarevlm.visual_merger_kl.eval_cell.v1"
        or result.get("input_lock_sha256") != content_sha(lock)
        or result.get("predictions_sha256") != sha_file(predictions)
        or any(result.get(key) != value for key, value in generation.summary(records, task).items())):
        raise ValueError("completed visual-merger full-task score/count/hash differs")
    canonical, proof = canonical_rows(records, task)
    return {"complete": True, "condition": condition, "seed": seed, "task": task, "rows": canonical,
        "lock": lock, "evaluation": result, "canonicalization": proof,
        "provenance": {"predictions_file": str(predictions), "predictions_sha256": sha_file(predictions),
            "evaluation_file": str(result_file), "evaluation_sha256": sha_file(result_file),
            "input_lock_file": str(place / "INPUT_LOCK.json"), "input_lock_sha256": sha_file(place / "INPUT_LOCK.json"),
            "training_result_sha256": binding["sha256"],
            "model_artifact_role":"frozen_language144_trainable_visual_merger4"}}
