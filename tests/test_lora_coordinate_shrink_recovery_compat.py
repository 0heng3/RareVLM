"""Pure stdlib synthetic contracts for the single permitted identity exception."""
from copy import deepcopy
import json
import math
from pathlib import Path
import pytest

from vlm_lora_coordinate_shrink_recovery_v1.compat import stable_binding_view,binding_identity_equal


def raw_binding(wall=.23652817774564028):
    # JSON-shaped interface fixture, not a generated model or performance result.
    producer={'schema_version':'producer_reference.v1','seed':42,
        'producer_result_file':'/opt/data/private/accepted/TRAIN_RESULT.json',
        'producer_result_file_sha256':'b'*64,'expected_final_master_sha256':'c'*64,
        'initial_canonical_master_sha256':'d'*64,'initial_adapter_files_sha256':{'adapter_model.safetensors':'e'*64},
        'source_binding_files_sha256':{'/opt/data/private/accepted/STUDY.json':'f'*64},
        'historical_training_updates':181,'new_training_optimizer_updates':0,'producer_was_shrunk':False}
    return {'schema_version':'rarevlm.lora_coordinate_shrink.adapter_binding.v1',
        'artifact_role':'coordinate_shrunk_candidate','path':'/opt/data/private/export/EXPORT_MANIFEST.json',
        'sha256':'a'*64,'condition':'lora_half32','seed':42,
        'producer_result_file':producer['producer_result_file'],'producer_result_file_sha256':'b'*64,
        'producer_reference':producer,
        'exported_fp32_tensor_sha256':'1'*64,'exported_bf16_tensor_sha256':'2'*64,
        'exported_bf16_round_fp32_tensor_sha256':'3'*64,
        'export_verification':{'schema_version':'verified_export.v1','status':'verified_cpu_export',
            'export_manifest_file':'/opt/data/private/export/EXPORT_MANIFEST.json','export_manifest_file_sha256':'a'*64,
            'adapter_dir':'/opt/data/private/export/adapter','adapter_files_sha256':{'adapter_model.safetensors':'4'*64,'adapter_config.json':'5'*64},
            'source_producer_result_file_sha256':'b'*64,'producer_reference':deepcopy(producer),
            'tensor_count':144,'parameter_elements':1843200,'alpha':.5,'r':8,'lora_alpha':16,
            'numeric_relationship_verified_by_torch_equal':True,'new_training_updates':0,'new_GPU_operations':0,
            'verify_CPU_wall_seconds':wall},
        'scope':'fixed0.5 factor-coordinate export','other_engineering':{'CPU_wall_seconds':12.,
            'verify_CPU_wall_seconds':7.,'events':['retained','exact']}}


def set_path(value,path,new):
    for key in path[:-1]:value=value[key]
    value[path[-1]]=new


def test_two_finite_real_wall_values_have_same_identity_without_mutating_raw_locks():
    saved=raw_binding();fresh=raw_binding(.947125)
    before_saved=json.dumps(saved,sort_keys=True);before_fresh=json.dumps(fresh,sort_keys=True)
    assert binding_identity_equal(saved,fresh)
    a=stable_binding_view(saved);b=stable_binding_view(fresh)
    assert a==b and 'verify_CPU_wall_seconds' not in a['export_verification']
    assert a['other_engineering']['verify_CPU_wall_seconds']==7.  # Exact path only.
    assert json.dumps(saved,sort_keys=True)==before_saved and json.dumps(fresh,sort_keys=True)==before_fresh
    a['export_verification']['adapter_files_sha256']['adapter_model.safetensors']='mutated view'
    a['producer_reference']['source_binding_files_sha256']['added']='view only'
    assert saved['export_verification']['adapter_files_sha256']['adapter_model.safetensors']=='4'*64
    assert 'added' not in saved['producer_reference']['source_binding_files_sha256']


@pytest.mark.parametrize('wall',[0,0.,1,1e-20,1e20])
def test_nonnegative_real_engineering_values_are_valid(wall):
    assert binding_identity_equal(raw_binding(wall),raw_binding(.2))


@pytest.mark.parametrize('bad',[True,False,None,'0.2',complex(1,0),float('nan'),float('inf'),-float('inf'),-.000001,10**10000],
    ids=['bool_true','bool_false','none','string','complex','nan','inf','negative_inf','negative','overflow_real'])
def test_invalid_wall_is_rejected_on_either_side(bad):
    invalid=raw_binding(bad)
    with pytest.raises(ValueError,match='finite nonnegative real'):stable_binding_view(invalid)
    with pytest.raises(ValueError):binding_identity_equal(raw_binding(),invalid)
    with pytest.raises(ValueError):binding_identity_equal(invalid,raw_binding())


@pytest.mark.parametrize('broken',[{}, {'export_verification':None},{'export_verification':[]},{'export_verification':{}}])
def test_missing_or_invalid_timing_container_is_never_treated_as_stable(broken):
    with pytest.raises(ValueError):stable_binding_view(broken)
    with pytest.raises(ValueError):binding_identity_equal(raw_binding(),broken)


def test_stripped_view_cannot_be_reused_as_raw_or_mutate_a_saved_lock():
    raw=raw_binding();view=stable_binding_view(raw)
    with pytest.raises(ValueError,match='missing'):binding_identity_equal(raw,view)
    with pytest.raises(ValueError,match='missing'):stable_binding_view(view)
    assert 'verify_CPU_wall_seconds' in raw['export_verification']


@pytest.mark.parametrize('path,value',[
    (('schema_version',),'other.v1'),(('artifact_role',),'unshrunk_producer'),
    (('path',),'/opt/data/private/other/EXPORT_MANIFEST.json'),(('sha256',),'0'*64),
    (('condition',),'teacher_kl32'),(('seed',),200),
    (('producer_result_file',),'/opt/data/private/other/TRAIN_RESULT.json'),(('producer_result_file_sha256',),'0'*64),
    (('producer_reference','initial_canonical_master_sha256'),'0'*64),
    (('producer_reference','expected_final_master_sha256'),'0'*64),
    (('producer_reference','source_binding_files_sha256','/opt/data/private/accepted/STUDY.json'),'0'*64),
    (('producer_reference','historical_training_updates'),180),
    (('exported_fp32_tensor_sha256',),'0'*64),(('exported_bf16_tensor_sha256',),'0'*64),
    (('exported_bf16_round_fp32_tensor_sha256',),'0'*64),
    (('export_verification','export_manifest_file_sha256'),'0'*64),
    (('export_verification','adapter_dir'),'/opt/data/private/raw_producer/adapter'),
    (('export_verification','adapter_files_sha256','adapter_model.safetensors'),'0'*64),
    (('export_verification','adapter_files_sha256','adapter_config.json'),'0'*64),
    (('export_verification','source_producer_result_file_sha256'),'0'*64),
    (('export_verification','producer_reference','seed'),200),
    (('export_verification','tensor_count'),143),(('export_verification','parameter_elements'),3686400),
    (('export_verification','alpha'),.25),(('export_verification','r'),16),(('export_verification','lora_alpha'),8),
    (('export_verification','numeric_relationship_verified_by_torch_equal'),False),
    (('export_verification','new_training_updates'),1),
    (('other_engineering','CPU_wall_seconds'),13.),
    (('other_engineering','verify_CPU_wall_seconds'),8.),
    (('scope',),'changed'),
])
def test_each_other_scientific_or_structural_change_is_rejected(path,value):
    saved=raw_binding();fresh=raw_binding(.8);set_path(fresh,path,value)
    assert binding_identity_equal(saved,fresh) is False


def test_extra_missing_types_and_nested_structure_remain_exact():
    saved=raw_binding()
    for mutate in [lambda v:v.update(extra='new'),lambda v:v.pop('scope'),
            lambda v:set_path(v,('producer_reference','producer_was_shrunk'),0),
            lambda v:set_path(v,('export_verification','new_training_updates'),False),
            lambda v:set_path(v,('export_verification','r'),8.),
            lambda v:set_path(v,('other_engineering','events'),('retained','exact')),
            lambda v:set_path(v,('other_engineering','events'),['exact','retained'])]:
        fresh=raw_binding(.5);mutate(fresh)
        assert binding_identity_equal(saved,fresh) is False
