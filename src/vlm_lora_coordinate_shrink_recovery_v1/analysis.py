"""One fixed coordinate-shrunk candidate; unshrunk KL is a historical reference."""
PARENT_UPSTREAM_COPY_PROVENANCE = {'source': 'src/vlm_object_replay_coverage/analysis.py', 'sha256': 'fa882eaf36958ea8648643a63dc0a7500c41fbd07fa30a6b313ef4c0bbd788d1', 'scope': 'independent adapted copy; frozen original globals and bytes are unchanged'}
COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_object_behavior_replay/analysis.py', 'sha256': '03cdf297b4cba2fbc1d39e8495a8cc9f7ea6c06d5e6d01a01e3fff08971f8f14', 'scope': 'explicit new namespace copy; original globals/bytes unchanged'}
COPY_PROVENANCE = {'source': '/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/src/vlm_generated_prefix_kl/analysis.py', 'sha256': '922327b024e364e846a45cd6c373591320136a7b8839594b79a25e58de07719c', 'scope': 'independent coordinate-shrink copy; original modules/globals are untouched'}
RECOVERY_COPY_PROVENANCE={'source': 'src/vlm_lora_coordinate_shrink/analysis.py', 'sha256': '28a7ef4b77eb8f8b8ecccea97f64c4fa715f5c4a357f2971cec67a43ef5a0706', 'scope': 'independent analysis copy; unchanged numeric/gate helpers with mandatory effective recovery provenance'}
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
import argparse

from vlm_object_eval.analysis import binary_contrast,confirm_gate as pope_gate
from vlm_object_kl_eval.analysis import (assert_pair,text_contrasts,balanced_contrast,screen_gate,
    external_gate,validate_settings)
from vlm_posttrain_eval.analysis import prepare_slices
from vlm_probe.evaluate import now,read_json,sha_file,write_json
from .contracts import (CANDIDATE,CONTROL,path,read_cell,source_hashes,verify_source_data,phase_admission,
    verify_context,recovery_fields,verify_recovery_fields)


def passing_pilot(file,config_file):
    verify_context(config_file)
    value=read_json(file);verify_recovery_fields(value,config_file)
    if (value.get('schema_version')!='rarevlm.lora_coordinate_shrink.screen.v1'
        or value.get('status')!='complete' or value.get('seeds')!=[42]
        or value.get('candidate')!=CANDIDATE or value.get('eligible_candidates')!=[CANDIDATE]
        or value.get('gate',{}).get('passed') is not True
        or value.get('config_sha256')!=sha_file(config_file) or value.get('source_sha256')!=source_hashes()):
        raise ValueError('only unchanged passing lora_half32 pilot may expand to seeds0/200')
    verify_source_data(value['source_data'])
    return value


def reuse_complete_analysis(destination,current,phase,prior,bindings):
    """Keep completed analysis bytes stable across complete-batch controller resumes."""
    if not destination.exists():return None
    saved=read_json(destination)
    if saved.get('status')!='complete':return None
    keys=('schema_version','stage','config_sha256','source_sha256','seeds','candidate','exploratory_control',
        'settings','source_data','canonicalization','control_can_be_selected','control_reference_only',
        'recovery_input_lock_path','recovery_input_lock_sha256','recovery_source_sha256')
    if any(saved.get(key)!=current.get(key) for key in keys):
        raise ValueError('completed analysis identity/input sources changed; no overwrite')
    if any(saved.get(key)!=value for key,value in bindings.items()):
        raise ValueError('completed analysis prior-phase raw bytes changed; no overwrite')
    verify_source_data(saved['source_data'])
    if phase=='screen':expected=screen_gate(saved['contrasts']['object'],saved['contrasts']['text'],saved['class_balanced'],replicated=len(saved['seeds'])==3)
    elif phase=='external':expected=external_gate(saved['contrasts']['external'],saved['class_balanced'],prior)
    else:expected=pope_gate(saved['contrasts']['pope'],{'passed':prior['gate']['passed']})
    if saved.get('gate')!=expected:raise ValueError('completed analysis original gate flags differ')
    return saved


def analyze(config_file,root,seeds=(42,),phase='screen',*,pilot_file=None,screen_file=None,external_file=None):
    config_file,root=Path(config_file).resolve(),Path(root).resolve()
    verify_context(config_file)
    config=read_json(config_file);settings=validate_settings(config)
    seeds=sorted(set(seeds))
    if seeds not in ([42],[0,42,200]) or phase not in ('screen','external','confirm'):
        raise ValueError('fixed pilot42 or full three-order-seed phases required')
    prior=None
    if phase=='screen' and seeds==[0,42,200]:
        pilot_file=Path(pilot_file or root/'analysis/pilot_seed42/screen/ANALYSIS.json')
        prior=passing_pilot(pilot_file,config_file)
    elif phase in ('external','confirm'):
        if seeds!=[0,42,200]:raise ValueError('confirmation requires all three lora_half32 order seeds')
        prior=phase_admission(screen_file if phase=='external' else external_file,config_file,
                              'screen' if phase=='external' else 'external')
    tasks=('object','text') if phase=='screen' else ('external',) if phase=='external' else ('pope',)
    cells={}
    for task in tasks:
        cells[f'initial_42_{task}']=read_cell(root,'initial',42,task,config_file,config)
        for seed in seeds:cells[f'{CANDIDATE}_{seed}_{task}']=read_cell(root,CANDIDATE,seed,task,config_file,config)
        if phase=='screen' and seeds==[42]:
            cells[f'{CONTROL}_42_{task}']=read_cell(root,CONTROL,42,task,config_file,config)
    destination=root/'analysis'/('pilot_seed42' if seeds==[42] else 'seeds_0_42_200')/phase/'ANALYSIS.json'
    result={'schema_version':f'rarevlm.lora_coordinate_shrink.{phase}.v1','status':'waiting_predictions',
        'stage':'pilot_order_seed42' if seeds==[42] else 'replicated_internal_screen' if phase=='screen' else phase+'_confirmation',
        **recovery_fields(config_file),'updated_at':now(),'config_sha256':sha_file(config_file),'source_sha256':source_hashes(),
        'seeds':seeds,'candidate':CANDIDATE,'exploratory_control':CONTROL,'settings':settings,
        'pending_cells':[key for key,cell in cells.items() if not cell['complete']],
        'initial_checkpoint':'one shared fixed SFT42; all four complete baselines are explicit historical references',
        'control_can_be_selected':False,'equal_compute_claim':False,'control_reference_only':True,
        'limitations':['pilot control comparison is single-order-seed exploratory; no control replication or selection',
            'four primary endpoints for lora_half32 only alpha=.05/4; not whole-history FWER',
            'image-cluster uncertainty conditional on checkpoints; seed means before resampling and sample SD separate',
            'public external/POPE pretraining overlap unknown; historical baselines are not new independent runs']}
    if result['pending_cells']:
        write_json(destination,result);return result,destination
    result.update(status='complete',source_data={key:cell['provenance'] for key,cell in cells.items()},
        canonicalization={key:cell['canonicalization'] for key,cell in cells.items()},contrasts={},summaries={})
    bindings={}
    if phase=='screen' and prior:
        bindings={'pilot_screen_path':str(pilot_file),'pilot_screen_sha256':sha_file(pilot_file)}
    elif phase=='external':bindings={'screen_analysis_path':str(screen_file),'screen_analysis_sha256':sha_file(screen_file)}
    elif phase=='confirm':bindings={'external_analysis_path':str(external_file),'external_analysis_sha256':sha_file(external_file)}
    saved=reuse_complete_analysis(destination,result,phase,prior,bindings)
    if saved is not None:return saved,destination
    features=None
    for task in tasks:
        initial=cells[f'initial_42_{task}'];candidates=[cells[f'{CANDIDATE}_{seed}_{task}'] for seed in seeds]
        for cell in candidates:assert_pair(initial,cell,task)
        if task=='text':
            slices=prepare_slices(path(config,'asset_root'),root/'analysis/text_slices')
            features=slices['features'];result['text_slice_lock_sha256']=slices['slice_lock_sha256']
            values=text_contrasts(initial['rows'],[cell['rows'] for cell in candidates],features)
        else:
            values,summary=binary_contrast(initial['rows'],[cell['rows'] for cell in candidates])
            result['summaries'][task]=summary
        result['contrasts'][task]=values
        if task in ('object','external'):
            result['class_balanced']=balanced_contrast(config,task,initial['rows'],[cell['rows'] for cell in candidates])
    if phase=='screen':
        result['gate']=screen_gate(result['contrasts']['object'],result['contrasts']['text'],
                                    result['class_balanced'],replicated=len(seeds)==3)
        result['eligible_candidates']=[CANDIDATE] if result['gate']['passed'] else []
        if seeds==[42]:
            exploratory={}
            for task in ('object','text'):
                initial,control,wide=(cells[f'{name}_42_{task}'] for name in ('initial',CONTROL,CANDIDATE))
                assert_pair(initial,control,task);assert_pair(control,wide,task)
                for label,left,right in ((f'{CONTROL}_minus_initial',initial,control),(f'{CANDIDATE}_minus_{CONTROL}',control,wide)):
                    exploratory.setdefault(label,{})[task]=(text_contrasts(left['rows'],[right['rows']],features)
                        if task=='text' else binary_contrast(left['rows'],[right['rows']])[0])
            result['exploratory_control_contrasts']={'contrasts':exploratory,'selected':False,
                'scope':'single-seed descriptive contrasts; no control eligibility, performance confirmation, or equal teacher-compute claim'}
        if prior:result.update(pilot_screen_path=str(pilot_file),pilot_screen_sha256=sha_file(pilot_file))
    elif phase=='external':
        result['gate']=external_gate(result['contrasts']['external'],result['class_balanced'],prior)
        result.update(screen_analysis_path=str(screen_file),screen_analysis_sha256=sha_file(screen_file))
    else:
        result['gate']=pope_gate(result['contrasts']['pope'],{'passed':prior['gate']['passed']})
        result.update(external_analysis_path=str(external_file),external_analysis_sha256=sha_file(external_file),
            confirmation_passed=result['gate']['passed'],goal_complete=False,root_independent_prediction_review_required=True)
    write_json(destination,result);return result,destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--phase',choices=('screen','external','confirm'),required=True)
    p.add_argument('--seeds',type=int,nargs='+',choices=(0,42,200),default=[42])
    for name in ('pilot-screen','screen-analysis','external-analysis'):p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    result,file=analyze(args.config,args.output_root,args.seeds,args.phase,pilot_file=args.pilot_screen,
                        screen_file=args.screen_analysis,external_file=args.external_analysis)
    print(result['status'],str(file),flush=True)


if __name__=='__main__':main()
