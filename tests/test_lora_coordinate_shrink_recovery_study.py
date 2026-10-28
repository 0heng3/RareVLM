"""Actual recovery reader and complete orchestration transitions, synthetic CPU only."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from vlm_lora_coordinate_shrink import CANDIDATE
from vlm_lora_coordinate_shrink_recovery_v1 import contracts,study,analysis
from vlm_probe.evaluate import sha_file,content_sha


def dump(file,value):
    file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value));return file


def binding(wall=1.):
    return {'sha256':'manifest','producer_result_file_sha256':'producer','producer_result_file':'/opt/data/private/synthetic/TRAIN_RESULT.json',
        'path':'/opt/data/private/synthetic/EXPORT_MANIFEST.json','exported_bf16_tensor_sha256':'native',
        'exported_bf16_round_fp32_tensor_sha256':'promoted',
        'export_verification':{'verify_CPU_wall_seconds':wall,'exported_fp32_tensor_sha256':'half'}}


@pytest.mark.parametrize('mutation',[None,'producer','native','missing_timing'])
def test_actual_read_cell_accepts_only_timing_change_and_preserves_original_bytes(tmp_path,monkeypatch,mutation):
    root=tmp_path/'output';config={'output_root':str(root),'model_root':str(tmp_path/'model')}
    config_file=dump(tmp_path/'config.json',config);place=root/'runs'/CANDIDATE/'seed42/object'
    fresh=binding(9.);saved=binding(.2);files={'adapter_config.json':'cfg','adapter_model.safetensors':'weights'}
    records=[{'question_id':1,'image_id':'i','score':1.},{'question_id':2,'image_id':'i','score':1.}]
    metadata={'questions':2,'unique_images':1};summary={'questions':2,'score_percent':100.}
    lock={**contracts.FIXED,**metadata,'condition':CANDIDATE,'seed':42,'task':'object',
        'schema_version':'rarevlm.lora_coordinate_shrink.eval_input.v1','study_config_sha256':sha_file(config_file),
        'study_config_content_sha256':content_sha(config),'source_sha256':{},'packages':{},'prompt_suffix':'suffix',
        'adapter_dir':str(tmp_path/'adapter'),'adapter_files_sha256':files,'training_binding':saved,
        'actual_native_bf16_tensor_sha256':'native','actual_native_bf16_round_fp32_tensor_sha256':'promoted'}
    input_file=dump(place/'INPUT_LOCK.json',lock);pred=place/'predictions.jsonl';pred.write_text(''.join(json.dumps(r)+'\n' for r in records))
    result=dump(place/'EVALUATION.json',{'schema_version':'rarevlm.lora_coordinate_shrink.eval_cell.v1','status':'completed',
        'input_lock_sha256':content_sha(lock),'predictions_sha256':sha_file(pred),**summary})
    original={p:p.read_bytes() for p in (input_file,pred,result)}
    monkeypatch.setattr(contracts,'verify_context',lambda *a:{})
    monkeypatch.setattr(contracts,'load_data',lambda *a:(records,{'i':{}},tmp_path,metadata))
    monkeypatch.setattr(contracts,'source_hashes',lambda:{})
    monkeypatch.setattr(contracts,'package_versions',lambda:{})
    monkeypatch.setattr(contracts,'adapter_binding',lambda *a,**k:(files,fresh))
    monkeypatch.setattr(contracts,'generation',SimpleNamespace(suffix=lambda task:'suffix',
        complete_cache=lambda *a,**k:(records,False),summary=lambda *a:summary))
    if mutation=='producer':fresh['producer_result_file_sha256']='different'
    elif mutation=='native':fresh['exported_bf16_tensor_sha256']='different'
    elif mutation=='missing_timing':fresh['export_verification'].pop('verify_CPU_wall_seconds')
    if mutation is None:
        cell=contracts.read_cell(root,CANDIDATE,42,'object',config_file,config,processor=object())
        assert cell['complete'] and cell['rows']==records
        assert cell['provenance']['training_result_sha256']=='producer'
        assert cell['provenance']['adapter_export_sha256']=='manifest'
    else:
        with pytest.raises(ValueError):contracts.read_cell(root,CANDIDATE,42,'object',config_file,config,processor=object())
    assert all(file.read_bytes()==data for file,data in original.items())


def test_new_text_without_mandatory_recovery_source_is_rejected(tmp_path,monkeypatch):
    config_file=dump(tmp_path/'config.json',{'output_root':str(tmp_path)})
    dump(tmp_path/'runs'/CANDIDATE/'seed42/text/EVALUATION.json',{})
    dump(tmp_path/'runs'/CANDIDATE/'seed42/text/INPUT_LOCK.json',{})
    monkeypatch.setattr(contracts,'verify_context',lambda *a:{})
    monkeypatch.setattr(contracts,'recovery_fields',lambda *a:{'recovery_input_lock_sha256':'required'})
    with pytest.raises(ValueError,match='mandatory effective recovery'):
        contracts.read_cell(tmp_path,CANDIDATE,42,'text',config_file,{},processor=object())


def test_resume_lock_keeps_original_raw_timing_content_and_rejects_any_other_field_change(tmp_path):
    file=dump(tmp_path/'INPUT_LOCK.json',{'source_sha256':{'f':'sha'},'training_binding':binding(.2)})
    original=file.read_bytes();fresh={'source_sha256':{'f':'sha'},'training_binding':binding(9.)}
    assert study.resume_input_lock(file,fresh)['training_binding']==binding(.2)
    assert file.read_bytes()==original and fresh['training_binding']==binding(9.)
    bad=deepcopy(fresh);bad['source_sha256']['f']='different'
    with pytest.raises(ValueError):study.resume_input_lock(file,bad)
    assert file.read_bytes()==original


def test_recovery_gpu_flow_preserves_original_native_loading_and_generation_calls():
    import inspect
    from vlm_lora_coordinate_shrink import evaluate as original
    old=ast.parse(inspect.getsource(original.main));new=ast.parse(inspect.getsource(study.evaluate_cell))
    def calls(tree,name):
        return [ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Call)
            and ((isinstance(node.func,ast.Name) and node.func.id==name)
                or isinstance(node.func,ast.Attribute) and node.func.attr==name)]
    for name in ('load_path','set_float32_matmul_precision','register_forward_pre_hook','processor_for','run_preflight','run_cell','tensor_hash'):
        assert calls(old,name)==calls(new,name),name


def test_pilot_worker_reuses_object_and_calls_only_text_under_distinct_identity(tmp_path,monkeypatch):
    root=tmp_path/'output';config_file=dump(tmp_path/'config.json',{'output_root':str(root),'model_root':str(tmp_path/'model')})
    original_lock=dump(root/'STUDY_INPUT_LOCK.json',{'original':'immutable'})
    recovery_lock=dump(root/'recovery_v1/RECOVERY_INPUT_LOCK.json',{'recovery':'immutable'})
    dump(root/'runs'/CANDIDATE/'seed42/object/EVALUATION.json',{'status':'completed'})
    monkeypatch.setattr(study,'verify_context',lambda *a:{'original_eval_gpu_uuid':'original-uuid'})
    monkeypatch.setattr(study,'recovery_fields',lambda *a:{'recovery_input_lock_sha256':sha_file(recovery_lock)})
    monkeypatch.setattr(study,'verify_study',lambda *a,**k:{})
    monkeypatch.setattr(study,'visible_gpu_binding',lambda:{'uuid':'original-uuid'})
    monkeypatch.setattr(study,'adapter_binding',lambda *a,**k:({},binding(.3)))
    calls=[]
    monkeypatch.setattr(study,'task_admission',lambda place,*a,**k:dump(place/'admissions/text.json',{'synthetic':True}))
    def invoke(place,name,artifacts,phase,command,preflight):
        assert phase=='text' and command[command.index('--task')+1]=='text'
        calls.append(command)
    monkeypatch.setattr(study,'subprocess_call',invoke)
    def read(output,condition,seed,task,*a):
        source={}
        for kind in ('predictions','evaluation','input_lock'):
            file=dump(tmp_path/'synthetic_artifacts'/task/(kind+'.json'),{'task':task})
            source[kind+'_file']=str(file);source[kind+'_sha256']=sha_file(file)
        return {'complete':True,'provenance':source}
    monkeypatch.setattr(study,'read_cell',read)
    args=SimpleNamespace(config=config_file,output_root=root,study_lock=original_lock,recovery_lock=recovery_lock,
        operation='eval',condition=CANDIDATE,seed=42,shard=None,screen_analysis=None,external_analysis=None)
    receipt=study.worker_run(args)
    assert len(calls)==1 and receipt['role']=='recovery_v1_eval_lora_half32_seed42'
    assert receipt['status']=='completed' and receipt['gpu_uuid']=='original-uuid'
    assert not (root/'workers').exists()
    assert (root/'recovery_v1/workers'/receipt['role']/'COMPLETE.json').exists()


def test_pilot_admission_requires_original_numeric_gpu_uuid(tmp_path,monkeypatch):
    config=dump(tmp_path/'config.json',{'output_root':str(tmp_path/'output')})
    monkeypatch.setattr(study,'verify_context',lambda *a:{'original_eval_gpu_uuid':'original-uuid'})
    with pytest.raises(ValueError,match='original numeric role GPU UUID'):
        study.task_admission(tmp_path/'worker',config,CANDIDATE,42,'text',{'uuid':'wrong'})
    with pytest.raises(ValueError,match='never be generated'):
        study.eval_arguments(config,tmp_path,CANDIDATE,42,'object',tmp_path/'adapter',tmp_path/'admission')
    with pytest.raises(ValueError):study.train_arguments(config,tmp_path,CANDIDATE,42)


def test_queue_pins_original_gpu_before_shared_admission_and_rejects_affinity_change(tmp_path,monkeypatch):
    config=dump(tmp_path/'config.json',{})
    monkeypatch.setattr(study,'verify_context',lambda *a:{})
    jobs=[{'key':'role','command':['synthetic'],'completion_marker':str(tmp_path/'marker.json')}]
    checks=[]
    def run(queue,jobs,identity):checks.append(json.loads((queue/'STATUS.json').read_text())['jobs']['role']['gpu_uuid'])
    monkeypatch.setattr(study,'run_queue',run)
    monkeypatch.setattr(study,'read_receipt',lambda *a:{'status':'completed'})
    study.queue_stage(tmp_path,'pilot42',jobs,{'original_eval_gpu_uuid':'original-uuid'},config)
    assert checks==['original-uuid']
    file=tmp_path/'recovery_v1/queues/pilot42/STATUS.json';value=json.loads(file.read_text());value['jobs']['role']['gpu_uuid']='wrong';dump(file,value)
    with pytest.raises(ValueError,match='pinned original'):
        study.queue_stage(tmp_path,'pilot42',jobs,{'original_eval_gpu_uuid':'original-uuid'},config)


def scenario(tmp_path,monkeypatch,*,failed=None,engineering=None):
    root=tmp_path/'output';root.mkdir();config={'output_root':str(root)};file=dump(tmp_path/'config.json',config)
    dump(root/'STUDY_INPUT_LOCK.json',{'original':True});events=[]
    monkeypatch.setattr(study,'prepare',lambda *a:(config,root,{'original_eval_gpu_uuid':'uuid'}))
    monkeypatch.setattr(study,'verify_context',lambda *a:{'original_eval_gpu_uuid':'uuid'})
    monkeypatch.setattr(study,'recovery_fields',lambda *a:{'recovery_input_lock_sha256':'recovery'})
    monkeypatch.setattr(study,'prepare_generation_inputs',lambda cfg,seed:events.append(('prompt',seed)))
    monkeypatch.setattr(study,'finalize_teacher_replies',lambda cfg,seed:events.append(('merge',seed)))
    monkeypatch.setattr(study,'prepare_producer_inputs',lambda file,cfg,seed:events.append(('prepare_inputs',seed)))
    monkeypatch.setattr(study,'ensure_export',lambda file,cfg,seed:events.append(('export',seed)))
    def queue(root,name,jobs,identity,config):
        items=[]
        for job in jobs:
            cmd=job['command'];get=lambda key:cmd[cmd.index(key)+1]
            items.append((get('--operation'),get('--condition'),int(get('--seed')),int(get('--shard')) if '--shard' in cmd else None))
        events.append(('queue',name,items))
        return [{'outcome':'engineering_rejected' if name==engineering else 'passed','role':job['key']} for job in jobs]
    monkeypatch.setattr(study,'queue_stage',queue)
    def analyze(file,output,seeds,phase,**kwargs):
        label='pilot' if list(seeds)==[42] else phase;events.append(('analyze',list(seeds),phase))
        return {'status':'complete','gate':{'passed':label!=failed}},root/'analysis'/(label+'.json')
    monkeypatch.setattr(study,'analyze',analyze)
    monkeypatch.setattr(study,'finish',lambda root,status,phase,analyses,receipts,started,config:{'status':status,'goal_complete':False})
    return file,root,events


def test_actual_negative_pilot_stops_without_new_producer_teacher_export_or_confirmation(tmp_path,monkeypatch):
    file,root,events=scenario(tmp_path,monkeypatch,failed='pilot');result=study.run_study(file)
    assert result=={'status':'completed_negative','goal_complete':False}
    assert [e for e in events if e[0]=='queue']==[('queue','pilot42',[('eval',CANDIDATE,42,None)])]
    assert not any(e[0] in ('prompt','prepare_inputs','export') for e in events)
    assert not (root/'STATUS.json').exists()


def test_actual_full_success_runs_original_0_200_then_three_full_external_and_pope(tmp_path,monkeypatch):
    file,root,events=scenario(tmp_path,monkeypatch);result=study.run_study(file)
    assert result=={'status':'confirmation_passed_pending_independent_review','goal_complete':False}
    queues={e[1]:e[2] for e in events if e[0]=='queue'}
    assert list(queues)==['pilot42','teacher_generation_seeds0_200','producer_replication','replication_evaluation','external_confirmation','pope_confirmation']
    assert queues['teacher_generation_seeds0_200']==[('teacher','initial',s,h) for s in (0,200) for h in (0,1)]
    assert queues['producer_replication']==[('producer',CANDIDATE,s,None) for s in (0,200)]
    assert [e for e in events if e[0]=='export']==[('export',0),('export',200)]
    assert [e for e in events if e[0]=='prompt']==[('prompt',0),('prompt',200)]
    for name,operation in [('external_confirmation','external'),('pope_confirmation','pope')]:
        assert queues[name]==[(operation,CANDIDATE,s,None) for s in (0,42,200)]
    assert [e for e in events if e[0]=='analyze']==[('analyze',[42],'screen'),('analyze',[0,42,200],'screen'),('analyze',[0,42,200],'external'),('analyze',[0,42,200],'confirm')]


@pytest.mark.parametrize('failed,count',[('screen',4),('external',5)])
def test_actual_later_negative_stops_full_scope_without_lowering_gate(tmp_path,monkeypatch,failed,count):
    file,root,events=scenario(tmp_path,monkeypatch,failed=failed);result=study.run_study(file)
    assert result['status']=='completed_negative' and len([e for e in events if e[0]=='queue'])==count


def test_actual_engineering_failure_is_separate_from_negative_scientific_gate(tmp_path,monkeypatch):
    file,root,events=scenario(tmp_path,monkeypatch,engineering='pilot42');result=study.run_study(file)
    assert result['status']=='engineering_rejected' and not any(e[0]=='analyze' for e in events)


def test_actual_completed_resume_does_not_dispatch_or_overwrite_original_failure(tmp_path,monkeypatch):
    file,root,events=scenario(tmp_path,monkeypatch)
    original=dump(root/'STATUS.json',{'status':'original_failure'});before=original.read_bytes()
    existing={'status':'completed_negative','goal_complete':False,'recovery_input_lock_sha256':'recovery',
        'analysis_files_sha256':{},'worker_receipts_sha256':{},'dynamic_input_files_sha256':{},'export_manifest_files_sha256':{}}
    dump(root/'recovery_v1/COMPLETION.json',existing)
    monkeypatch.setattr(study,'verify_recovery_fields',lambda value,*a:value)
    assert study.run_study(file)==existing and events==[] and original.read_bytes()==before


def test_completed_analysis_resume_rejects_changed_recovery_source_before_any_resampling(tmp_path,monkeypatch):
    current={'status':'complete','seeds':[42],'recovery_input_lock_sha256':'old','source_data':{},
        'contrasts':{'object':[],'text':[]},'class_balanced':{},'gate':{'passed':False}}
    file=dump(tmp_path/'ANALYSIS.json',current);changed={**current,'recovery_input_lock_sha256':'changed'}
    with pytest.raises(ValueError,match='completed analysis identity'):
        analysis.reuse_complete_analysis(file,changed,'screen',None,{})
