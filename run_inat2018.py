"""Frozen-protocol iNaturalist 2018 ZS/CE/LA and fixed-representation controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from inat2018_manifest import NUM_CLASSES, load_manifest
from methods import PrototypeClassifier, ResidualAdapter, build_loss
from run_imagenet_lt import (EXPECTED_CLIP_WEIGHT_SHA256, classify_metrics,
    evaluate, load_bank, save_json, sha256_file)

ALPHAS = (0., .25, .5, .75, 1., 1.5, 2.)


def save_predictions(path, predictions, targets):
    tmp = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp, predictions=predictions, targets=targets.cpu().numpy().astype(np.uint16))
    tmp.replace(path)


def train_one(seed, method, classifier, banks, manifest, metadata, output_dir):
    run_id = f'inat2018_seed{seed}_{method}'
    result_path = output_dir/f'{run_id}.json'
    prediction_path = output_dir/f'{run_id}_pred.npz'
    checkpoint_path = output_dir/f'{run_id}_adapter.pt'
    config = {'name':'AdamW', 'lr':.001, 'weight_decay':.01, 'batch_size':256, 'epochs':20}
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (existing['manifest_sha256'] != manifest['manifest_sha256'] or existing['bank_metadata'] != metadata
                or existing['optimizer'] != config or existing['train_seed'] != seed or existing['method'] != method
                or existing['prediction_sha256'] != sha256_file(prediction_path)
                or existing['checkpoint_sha256'] != sha256_file(checkpoint_path)):
            raise ValueError(f'existing run provenance mismatch: {run_id}')
        print('reusing',run_id,flush=True)
        return existing
    train_x,train_y = banks['train']
    val_x,val_y = banks['val']
    torch.manual_seed(seed)
    adapter = ResidualAdapter(512,64).to(train_x.device)
    initial_hash=hashlib.sha256()
    for name,value in adapter.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(value.cpu().numpy().tobytes())
    initialization_sha256=initial_hash.hexdigest()
    criterion = (build_loss('ce') if method == 'ce' else build_loss('la',manifest['class_counts'],tau=1.)).to(train_x.device)
    optimizer = torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.01)
    rng = torch.Generator().manual_seed(seed+100000)
    best_oa,best_epoch,best_state = -math.inf,None,None
    history=[]
    started=time.monotonic()
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in adapter.parameters()}
    assert not any(p.requires_grad for p in classifier.parameters())
    for epoch in range(1,21):
        adapter.train()
        order=torch.randperm(len(train_x),generator=rng)
        order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
        loss_sum=0.
        for start in range(0,len(order),256):
            take=order[start:start+256].to(train_x.device)
            loss=criterion(classifier(adapter(train_x[take])),train_y[take])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite train loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(take)
        adapter.eval()
        with torch.inference_mode():
            logits=classifier(adapter(val_x))
            val_oa=float(logits.argmax(1).eq(val_y).float().mean())
            val_nll=float(F.cross_entropy(logits,val_y))
            del logits
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train_x),'val_oa':val_oa,'val_nll':val_nll,'sampler_order_sha256':order_hash})
        if val_oa > best_oa:
            best_oa,best_epoch=val_oa,epoch
            best_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        print(f'{run_id} epoch={epoch} loss={loss_sum/len(train_x):.4f} val={val_oa:.4f} best={best_oa:.4f}@{best_epoch}',flush=True)
    adapter.load_state_dict(best_state)
    adapter.eval()
    metrics,predictions=evaluate(adapter,classifier,*banks['test'],manifest['groups'])
    torch.save({'adapter_state_dict':best_state,'manifest_sha256':manifest['manifest_sha256'],'run_id':run_id},checkpoint_path)
    save_predictions(prediction_path,predictions,banks['test'][1])
    result={'run_id':run_id,'dataset':'iNaturalist 2018','manifest_sha256':manifest['manifest_sha256'],
        'method':method,'train_seed':seed,'initialization_sha256':initialization_sha256,'model':'fixed B/32 text prototypes + LN/GELU residual adapter 512-64-512',
        'loss':{} if method=='ce' else {'tau':1.},'optimizer':config,
        'selection':{'criterion':'highest internal balanced-validation OA; earliest tie','best_epoch':best_epoch,'best_val_oa':best_oa},
        'test_split_semantics':manifest['split_semantics']['test'],'test':metrics,'history':history,
        'bank_metadata':metadata,'duration_seconds':round(time.monotonic()-started,2),
        'checkpoint_sha256':sha256_file(checkpoint_path),'prediction_sha256':sha256_file(prediction_path)}
    save_json(result_path,result)
    print(run_id,'final OA/Few',metrics['oa'],metrics['groups']['few']['accuracy'],flush=True)
    return result


@torch.inference_mode()
def run_controls(seed,method,classifier,banks,manifest,metadata,output_dir):
    run_id=f'inat2018_seed{seed}_{method}'
    checkpoint_path=output_dir/f'{run_id}_adapter.pt'
    result_path=output_dir/f'{run_id}.json'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('checkpoint manifest mismatch')
    adapter=ResidualAdapter(512,64).to(banks['train'][0].device).eval()
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    prior=torch.tensor(manifest['class_counts'],device=banks['train'][0].device,dtype=torch.float64)
    prior/=prior.sum()
    log_prior=prior.log().float()
    source_hash=sha256_file(checkpoint_path)
    main_result=json.loads(result_path.read_text())
    if main_result['checkpoint_sha256'] != source_hash:
        raise ValueError('checkpoint changed after selection')
    control_dir=output_dir/'controls'
    control_dir.mkdir(exist_ok=True)
    val_logits=classifier(adapter(banks['val'][0]))
    curve=[{'alpha':a,'val_oa':float((val_logits-a*log_prior).argmax(1).eq(banks['val'][1]).float().mean())} for a in ALPHAS]
    selected=max(curve,key=lambda row:(row['val_oa'],-row['alpha']))['alpha']
    del val_logits
    if method=='ce':
        selection={'alpha':selected,'curve':curve,'source':'internal train-held-out validation only',
                   'source_checkpoint_sha256':source_hash,'manifest_sha256':manifest['manifest_sha256']}
        selection_path=control_dir/f'{run_id}_VALIDATION_SELECTION.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError('existing validation selection differs')
        if not selection_path.exists():
            save_json(selection_path,selection)
    train_x=banks['train'][0]
    q=torch.zeros(NUM_CLASSES,device=train_x.device,dtype=torch.float64)
    for start in range(0,len(train_x),512):
        logits=classifier(adapter(train_x[start:start+512]))
        if method=='la':
            logits=logits+log_prior
        q+=logits.double().softmax(1).sum(0)
    q/=len(train_x)
    if method=='la':
        q=q/prior
        q/=q.sum()
    if not bool(torch.isfinite(q).all()) or bool((q<=0).any()):
        raise FloatingPointError('invalid effective prior')
    prior_path=control_dir/f'{run_id}_effective_prior.npz'
    np.savez_compressed(prior_path,effective_prior=q.cpu().numpy(),empirical_prior=prior.cpu().numpy())
    q_np,p_np=q.cpu().numpy(),prior.cpu().numpy()
    diagnostics={'pearson_log_prior':float(np.corrcoef(np.log(q_np),np.log(p_np))[0,1]),
        'spearman_prior':float(spearmanr(q_np,p_np).statistic),'total_variation':float(.5*np.abs(q_np-p_np).sum()),
        'prior_source':'all actual train feature predictions; no val/test estimation','temperature':'original CLIP scale; float64 softmax',
        'target_prior':'uniform 1/8142','prior_sha256':sha256_file(prior_path)}
    test_x,test_y=banks['test']
    adapted=adapter(test_x)
    logits=classifier(adapted)
    controls={'p2p_train_only':-(q.log().float())-math.log(NUM_CLASSES)}
    if method=='ce':
        controls={'empirical_alpha1':-log_prior,'val_alpha':-selected*log_prior,**controls}
    results={}
    for name,offset in controls.items():
        metrics,predictions=classify_metrics(logits+offset,adapted,test_x,test_y,classifier.text_prototypes,manifest['groups'])
        prediction_path=control_dir/f'{run_id}_{name}_pred.npz'
        save_predictions(prediction_path,predictions,test_y)
        base={'run_id':f'{run_id}_{name}','method':name,'source_method':method,'train_seed':seed,
            'manifest_sha256':manifest['manifest_sha256'],'source_checkpoint_sha256':source_hash,
            'test_split_semantics':manifest['split_semantics']['test'],'test':metrics,'prediction_sha256':sha256_file(prediction_path),
            'bank_metadata':metadata,'prior_diagnostics':diagnostics if name=='p2p_train_only' else None,
            'selection':{'alpha':selected,'curve':curve,'source':'internal train-held-out validation only'} if name=='val_alpha' else {'fixed_strength':1.},
            'representation_updated':False}
        save_json(control_dir/f'{run_id}_{name}.json',base)
        results[name]=base
        print(run_id,name,'OA/Few',metrics['oa'],metrics['groups']['few']['accuracy'],flush=True)
    if source_hash != sha256_file(checkpoint_path):
        raise ValueError('control changed source checkpoint')
    return results


def metric_row(metrics):
    return {'oa':metrics['oa'],**{k:metrics['groups'][k]['accuracy'] for k in ('many','medium','few')},
            'macro_f1':metrics['macro_f1'],'nll':metrics['nll'],'ece15':metrics['ece15']}


def summarize(zero,runs,controls,manifest,output_dir,seeds):
    rows={'zero_shot':[metric_row(zero['test'])]}
    for method in ('ce','la'):
        rows[method]=[metric_row(runs[(s,method)]['test']) for s in seeds]
    for name in ('empirical_alpha1','val_alpha','p2p_train_only'):
        rows['ce_'+name]=[metric_row(controls[(s,'ce')][name]['test']) for s in seeds]
    rows['la_p2p_train_only']=[metric_row(controls[(s,'la')]['p2p_train_only']['test']) for s in seeds]
    summary={'manifest_sha256':manifest['manifest_sha256'],'seeds':seeds,'test_split_semantics':manifest['split_semantics']['test'],
        'split_sizes':{k:len(v) for k,v in manifest['splits'].items()},'group_classes':{k:len(v) for k,v in manifest['groups'].items()},
        'metrics':{name:{k:{'mean':float(np.mean([r[k] for r in values])), 'seed_std':float(np.std([r[k] for r in values],ddof=1)) if len(values)>1 else None} for k in values[0]} for name,values in rows.items()},
        'selected_alpha':{str(s):controls[(s,'ce')]['val_alpha']['selection']['alpha'] for s in seeds}}
    contrasts={}
    pairs=[('ce','zero_shot'),('la','ce'),('ce_empirical_alpha1','ce'),('ce_val_alpha','ce'),('ce_p2p_train_only','ce'),('la','ce_val_alpha'),('la','ce_p2p_train_only')]
    for a,b in pairs:
        contrasts[f'{a}_minus_{b}']={k:100*(summary['metrics'][a][k]['mean']-summary['metrics'][b][k]['mean']) for k in ('oa','many','medium','few')}
    summary['contrasts_pp']=contrasts
    summary['directions']={}
    for i,s in enumerate(seeds):
        if runs[(s,'ce')]['initialization_sha256'] != runs[(s,'la')]['initialization_sha256']:
            raise ValueError('CE/LA initialization differs')
        ce,la,z=rows['ce'][i],rows['la'][i],rows['zero_shot'][0]
        summary['directions'][str(s)]={'A':ce['many']>z['many'] and ce['few']<z['few'],
            'B':la['oa']>ce['oa'] and la['few']>ce['few'],
            'C':{name:rows['ce_'+name][i]['oa']>ce['oa'] and rows['ce_'+name][i]['few']>ce['few'] for name in ('empirical_alpha1','val_alpha','p2p_train_only')}}
        if [r['sampler_order_sha256'] for r in runs[(s,'ce')]['history']] != [r['sampler_order_sha256'] for r in runs[(s,'la')]['history']]:
            raise ValueError('CE/LA sampler streams differ')
    summary['paired_sampler_verified']=True
    save_json(output_dir/'summary.json',summary)
    lines=['# RareVLM iNaturalist 2018：冻结协议结果','',
        '官方 val 为锁定的最终 top-1 研究评估，训练内每类 1 张仅用于选点。三种子共享同一评估图像；科学名单模板与每类仅 3 张评估图的限制见 项目 docs/PROTOCOL.md。','',
        '| 方法（三种子均值） | OA | Many | Medium | Few | Macro-F1 | NLL | ECE-15 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,values in summary['metrics'].items():
        values={k:v['mean'] for k,v in values.items()}
        lines.append('| '+name+' | '+' | '.join(f"{values[k]*(1 if k=='nll' else 100):.2f}" for k in ('oa','many','medium','few','macro_f1','nll','ece15'))+' |')
    lines+=['','## 预设方向（逐种子）','',json.dumps(summary['directions'],ensure_ascii=False,indent=2),'',
        '内部 val 选择 α：'+json.dumps(summary['selected_alpha']), '',
        '固定 CE 校正恢复量描述决策干预可达到的净准确率变化；不能当作纯决策偏置的因果占比，不能给未运行的复杂优化策略排序。', '',
        '机器结果和逐图预测：`outputs/inat2018_v1/main/`；特征 provenance：`outputs/inat2018_v1/features/metadata.json`。']
    # Machine completion artifact stays inside output directory; the reviewed
    # project report is written separately after all checks pass.
    (output_dir/'RESULTS.generated.md').write_text('\n'.join(lines)+'\n')
    print('complete summary',json.dumps(summary['directions']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--feature-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--train-seeds',type=int,nargs='+',default=[0,42,200])
    args=parser.parse_args()
    if args.train_seeds != [0,42,200]:
        raise ValueError('frozen protocol requires seeds 0,42,200')
    torch.set_num_threads(4)
    manifest=load_manifest(args.manifest)
    device=torch.device(args.device)
    banks={s:load_bank(args.feature_dir/f'{s}.pt',s,manifest['manifest_sha256'],manifest['splits'][s],device) for s in ('train','val','test')}
    metadata=json.loads((args.feature_dir/'metadata.json').read_text())
    proto=torch.load(args.feature_dir/'text_prototypes.pt',map_location='cpu',weights_only=True)
    if (metadata['manifest_sha256'] != manifest['manifest_sha256'] or proto['manifest_sha256'] != manifest['manifest_sha256']
        or metadata['model_weight_sha256'] != EXPECTED_CLIP_WEIGHT_SHA256 or proto['model_weight_sha256'] != EXPECTED_CLIP_WEIGHT_SHA256
        or proto['class_names'] != manifest['class_names'] or metadata['class_names'] != manifest['class_names']
        or proto['prompt_template'] != 'a photo of a {}.' or metadata['prompt_template'] != 'a photo of a {}.'
        or proto['text_prototypes'].shape != (NUM_CLASSES,512) or metadata['train_augmentation'] or metadata['tta']):
        raise ValueError('feature/prototype provenance mismatch')
    metadata['feature_artifact_sha256']={n:sha256_file(args.feature_dir/n) for n in ('train.pt','val.pt','test.pt','text_prototypes.pt')}
    classifier=PrototypeClassifier(proto['text_prototypes'],proto['logit_scale']).to(device).eval()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    zero_path=args.output_dir/'zero_shot.json'
    zero_pred=args.output_dir/'zero_shot_pred.npz'
    if zero_path.exists():
        zero=json.loads(zero_path.read_text())
        if zero['manifest_sha256'] != manifest['manifest_sha256'] or zero['bank_metadata'] != metadata or zero['prediction_sha256'] != sha256_file(zero_pred):
            raise ValueError('zero-shot provenance mismatch')
    else:
        metrics,predictions=evaluate(None,classifier,*banks['test'],manifest['groups'])
        save_predictions(zero_pred,predictions,banks['test'][1])
        zero={'method':'zero_shot','manifest_sha256':manifest['manifest_sha256'],'bank_metadata':metadata,
            'test_split_semantics':manifest['split_semantics']['test'],'test':metrics,'prediction_sha256':sha256_file(zero_pred)}
        save_json(zero_path,zero)
        print('zero-shot OA/Few',metrics['oa'],metrics['groups']['few']['accuracy'],flush=True)
    runs={}; controls={}
    for seed in args.train_seeds:
        for method in ('ce','la'):
            runs[(seed,method)]=train_one(seed,method,classifier,banks,manifest,metadata,args.output_dir)
            controls[(seed,method)]=run_controls(seed,method,classifier,banks,manifest,metadata,args.output_dir)
    summarize(zero,runs,controls,manifest,args.output_dir,args.train_seeds)


if __name__=='__main__':
    main()
