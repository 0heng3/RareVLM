"""Export completed experiment JSON records; never train or select parameters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

SEEDS = (0, 42, 200)
METRICS = ('oa', 'many', 'medium', 'few', 'macro_f1', 'nll', 'ece15')
METHODS = ('zero_shot', 'ce', 'la', 'ce_alpha1', 'ce_val_alpha',
           'ce_p2p_train_only', 'la_p2p_train_only')


def sources():
    def paths(prefix, stem):
        return [f'{prefix}/{stem.format(seed=s)}.json' for s in SEEDS]
    b32 = 'imagenet_lt_v1'
    b16 = 'imagenet_lt_b16_v1'
    online = 'imagenet_lt_p2_online_aug_v1'
    inat = 'inat2018_v1/main'
    return [
        {'id': 'imagenet_lt_b32', 'dataset': 'ImageNet-LT', 'checkpoint': 'ViT-B/32',
         'view': 'fixed_center_crop', 'manifest': b32+'/manifest.json',
         'metadata': b32+'/features/metadata.json', 'protocol': 'ImageNet-LT B/32 V3',
         'records': {
             'zero_shot': [b32+'/main/zero_shot.json'],
             'ce': paths(b32+'/main', 'imagenet_lt_seed{seed}_ce'),
             'la': paths(b32+'/main', 'imagenet_lt_seed{seed}_la'),
             'ce_alpha1': paths(b32+'/analysis/fixed_prior_controls', 'imagenet_lt_seed{seed}_ce_fixed1'),
             'ce_val_alpha': paths(b32+'/posthoc', 'imagenet_lt_seed{seed}_ce_posthoc'),
             'ce_p2p_train_only': paths(b32+'/analysis/p05_effective_prior', 'imagenet_lt_seed{seed}_ce_p2p'),
             'la_p2p_train_only': paths(b32+'/analysis/p05_effective_prior', 'imagenet_lt_seed{seed}_la_p2p')}},
        {'id': 'imagenet_lt_b16', 'dataset': 'ImageNet-LT', 'checkpoint': 'ViT-B/16',
         'view': 'fixed_center_crop', 'manifest': b32+'/manifest.json',
         'metadata': b16+'/features/metadata.json', 'protocol': 'P1 complete-checkpoint v1',
         'records': {
             'zero_shot': [b16+'/main/zero_shot.json'],
             'ce': paths(b16+'/main', 'imagenet_lt_b16_seed{seed}_ce'),
             'la': paths(b16+'/main', 'imagenet_lt_b16_seed{seed}_la'),
             'ce_alpha1': paths(b16+'/posthoc', 'imagenet_lt_b16_seed{seed}_ce_fixed1'),
             'ce_val_alpha': paths(b16+'/posthoc', 'imagenet_lt_b16_seed{seed}_ce_posthoc'),
             'ce_p2p_train_only': paths(b16+'/analysis/p1_train_only_p2p', 'imagenet_lt_b16_seed{seed}_ce_p2p'),
             'la_p2p_train_only': paths(b16+'/analysis/p1_train_only_p2p', 'imagenet_lt_b16_seed{seed}_la_p2p')}},
        {'id': 'imagenet_lt_b32_online', 'dataset': 'ImageNet-LT', 'checkpoint': 'ViT-B/32',
         'view': 'online_rrc_hflip', 'manifest': b32+'/manifest.json',
         'metadata': b32+'/features/metadata.json', 'protocol': 'P2 online-view v1',
         'records': {
             'zero_shot': [b32+'/main/zero_shot.json'],
             'ce': paths(online, 'imagenet_lt_seed{seed}_aug_ce'),
             'la': paths(online, 'imagenet_lt_seed{seed}_aug_la'),
             'ce_alpha1': paths(online+'/analysis/controls', 'imagenet_lt_seed{seed}_aug_ce_empirical_alpha1'),
             'ce_val_alpha': paths(online+'/analysis/controls', 'imagenet_lt_seed{seed}_aug_ce_empirical_val_alpha'),
             'ce_p2p_train_only': paths(online+'/analysis/controls', 'imagenet_lt_seed{seed}_aug_ce_p2p_alpha1'),
             'la_p2p_train_only': paths(online+'/analysis/controls', 'imagenet_lt_seed{seed}_aug_la_p2p_alpha1')}},
        {'id': 'inat2018_b32', 'dataset': 'iNaturalist 2018', 'checkpoint': 'ViT-B/32',
         'view': 'fixed_center_crop', 'manifest': 'inat2018_v1/manifest.json',
         'metadata': 'inat2018_v1/features/metadata.json', 'protocol': 'iNaturalist 2018 v1',
         'records': {
             'zero_shot': [inat+'/zero_shot.json'],
             'ce': paths(inat, 'inat2018_seed{seed}_ce'),
             'la': paths(inat, 'inat2018_seed{seed}_la'),
             'ce_alpha1': paths(inat+'/controls', 'inat2018_seed{seed}_ce_empirical_alpha1'),
             'ce_val_alpha': paths(inat+'/controls', 'inat2018_seed{seed}_ce_val_alpha'),
             'ce_p2p_train_only': paths(inat+'/controls', 'inat2018_seed{seed}_ce_p2p_train_only'),
             'la_p2p_train_only': paths(inat+'/controls', 'inat2018_seed{seed}_la_p2p_train_only')}}]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    seed_rows, main_rows, settings = [], [], {}
    for setting in sources():
        manifest = json.loads((archive/setting['manifest']).read_text())
        metadata = json.loads((archive/setting['metadata']).read_text())
        source_records = []
        for method in METHODS:
            rows = []
            for i, relative in enumerate(setting['records'][method]):
                path = archive/relative
                record = json.loads(path.read_text())
                if record['manifest_sha256'] != manifest['manifest_sha256']:
                    raise ValueError(f'manifest mismatch in {relative}')
                metrics = record['test']
                seed = '' if method == 'zero_shot' else SEEDS[i]
                selection = record.get('selection') or {}
                alpha = ''
                if method == 'ce_val_alpha':
                    for value in (record.get('selected_alpha'), record.get('alpha'),
                                  selection.get('alpha'), selection.get('selected_alpha')):
                        if value is not None:
                            alpha = float(value)
                            break
                    if alpha == '': raise ValueError(f'no selected alpha in {relative}')
                elif method not in ('zero_shot', 'ce', 'la'):
                    alpha = 1.0
                row = {'setting': setting['id'], 'dataset': setting['dataset'],
                       'checkpoint': setting['checkpoint'],
                       'training_view': 'not_trained' if method == 'zero_shot' else setting['view'],
                       'setting_training_view': setting['view'], 'method': method, 'seed': seed,
                       'alpha': alpha, 'selected_epoch': selection.get('best_epoch', selection.get('ce_best_epoch', '')),
                       'source_artifact': relative, 'source_sha256': sha(path),
                       'manifest_sha256': manifest['manifest_sha256']}
                for key in METRICS:
                    value = metrics['groups'][key]['accuracy'] if key in ('many','medium','few') else metrics[key]
                    row[key] = float(value)*(1 if key == 'nll' else 100)
                rows.append(row); seed_rows.append(row)
                source_records.append({'path': relative, 'sha256': row['source_sha256'],
                    'method': method, 'seed': seed,
                    'checkpoint_sha256': record.get('checkpoint_sha256', record.get('source_checkpoint_sha256', record.get('ce_checkpoint_sha256'))),
                    'prediction_sha256': record.get('prediction_sha256'), 'alpha': alpha})
            result = {k: rows[0][k] for k in ('setting','dataset','checkpoint','training_view','setting_training_view','method')}
            result.update({'n_runs':len(rows), 'train_seeds': ';'.join(str(r['seed']) for r in rows if r['seed'] != ''),
                           'alpha_by_seed': ';'.join(f"{r['seed']}:{r['alpha']:g}" for r in rows if r['alpha'] != '')})
            for key in METRICS:
                values = [r[key] for r in rows]
                result[key] = statistics.mean(values)
                result[key+'_sd'] = statistics.stdev(values) if len(values)>1 else ''
            main_rows.append(result)
        settings[setting['id']] = {
            'dataset':setting['dataset'], 'checkpoint':setting['checkpoint'],
            'training_view':setting['view'], 'protocol_version':setting['protocol'],
            'model_id':metadata['model'], 'model_revision':metadata.get('model_revision') or metadata['snapshot'],
            'model_weight_sha256':metadata['model_weight_sha256'],
            'manifest_sha256':manifest['manifest_sha256'], 'manifest_file_sha256':sha(archive/setting['manifest']),
            'split_sizes':metadata['split_sizes'], 'ordered_split_records_sha256':metadata['split_records_sha256'],
            'source_annotation_or_list_sha256':metadata.get('source_sha256') or metadata.get('split_source_sha256'),
            'prompt':metadata['prompt_template'], 'class_count':len(manifest['class_counts']),
            'frequency_group_class_counts':{k:len(v) for k,v in manifest['groups'].items()},
            'train_seeds':list(SEEDS), 'evaluation_tta':False,
            'final_evaluation_split':'official val, top-1 research evaluation' if setting['dataset']=='iNaturalist 2018' else 'official ImageNet-LT test',
            'zero_shot_reused_from':'imagenet_lt_b32' if setting['id']=='imagenet_lt_b32_online' else None,
            'source_result_artifacts':source_records}
    for name, rows in [('seed_results.csv',seed_rows),('main_results.csv',main_rows)]:
        with (destination/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:
                writer.writerow({k:f'{v:.10f}' if isinstance(v,float) else v for k,v in row.items()})
    provenance = {'schema_version':'rarevlm.public_results.v1', 'exported_on':'2026-10-01',
        'metric_units':{k:'NLL' if k=='nll' else 'percent' for k in METRICS},
        'aggregation':'arithmetic mean of the three completed training seeds; ZS is deterministic and appears once per setting',
        'uncertainty':'sample SD across training seeds (ddof=1), not a confidence interval or independent-data guarantee',
        'selected_runs_excluded':0, 'new_training_runs':0, 'settings':settings,
        'public_csv_sha256':{name:sha(destination/name) for name in ('main_results.csv','seed_results.csv')}}
    (destination/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(f'exported {len(main_rows)} aggregate rows and {len(seed_rows)} setting/seed rows; no training')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,required=True,help='directory containing the completed outputs/* experiment folders')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    args=p.parse_args();export(args.archive_root,args.output_dir)


if __name__=='__main__':main()
