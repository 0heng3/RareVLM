"""Extract immutable B/32 image features and official species-name prototypes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from extract_features import DEFAULT_SNAPSHOT, PROMPT, sha256_file
from extract_imagenet_lt_features import (ManifestImageDataset, atomic_json_save,
    atomic_torch_save, build_transform, content_sha256, encode_images,
    validate_existing_split)
from inat2018_manifest import NUM_CLASSES, load_manifest
from run_imagenet_lt import EXPECTED_CLIP_WEIGHT_SHA256


def encode_text(model, tokenizer, names, device):
    batches=[]
    with torch.inference_mode():
        for start in range(0,len(names),128):
            tokens=tokenizer([PROMPT.format(n) for n in names[start:start+128]],padding=True,truncation=True,return_tensors='pt')
            tokens={k:v.to(device) for k,v in tokens.items()}
            batches.append(F.normalize(model.get_text_features(**tokens).float(),dim=-1).cpu())
        scale=float(model.logit_scale.float().exp())
    prototypes=torch.cat(batches)
    if prototypes.shape != (len(names),512) or not torch.isfinite(prototypes).all():
        raise ValueError('invalid iNaturalist text prototypes')
    return prototypes,scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = load_manifest(args.manifest)
    transform, preproc = build_transform(args.model_snapshot)
    image_root = Path(manifest['image_root'])
    if args.check_only:
        for split, records in manifest['splits'].items():
            dataset = ManifestImageDataset(records, image_root, transform)
            for index in [0, len(dataset)//2, len(dataset)-1]:
                x = dataset[index]
                if x.shape != (3,224,224) or not torch.isfinite(x).all():
                    raise ValueError('invalid preprocessed image')
            print(split, 'decoded 3 samples;', len(records), 'images')
        return
    weight_hash = sha256_file(args.model_snapshot/'pytorch_model.bin')
    if weight_hash != EXPECTED_CLIP_WEIGHT_SHA256:
        raise ValueError('B/32 checkpoint differs from ImageNet-LT')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = manifest['class_names']
    metadata = {'schema_version': 'rarevlm.inat2018.features.v1', 'model': 'openai/clip-vit-base-patch32',
        'snapshot': args.model_snapshot.name, 'model_weight_sha256': weight_hash,
        'manifest_sha256': manifest['manifest_sha256'], 'source_sha256': manifest['source_sha256'],
        'split_records_sha256': {k:content_sha256(v) for k,v in manifest['splits'].items()},
        'split_sizes': {k:len(v) for k,v in manifest['splits'].items()},
        'class_names': names, 'class_names_sha256': content_sha256(names),
        'prompt_template': PROMPT, 'prompt_name_rule': 'unaltered official scientific species name',
        'image_preprocessing': preproc, 'train_augmentation': False, 'tta': False,
        'feature_dtype': 'float16 stored; float32 for adapter',
        'model_config_sha256': sha256_file(args.model_snapshot/'config.json'),
        'preprocessor_sha256': sha256_file(args.model_snapshot/'preprocessor_config.json')}
    meta_path = args.output_dir/'metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError('existing cache metadata mismatch')
    prototype_path = args.output_dir/'text_prototypes.pt'
    if prototype_path.exists():
        p = torch.load(prototype_path, map_location='cpu', weights_only=True)
        if (p['manifest_sha256'] != manifest['manifest_sha256'] or p['class_names'] != names
                or p['model_weight_sha256'] != weight_hash or p['prompt_template'] != PROMPT
                or p['text_prototypes'].shape != (NUM_CLASSES,512) or not torch.isfinite(p['text_prototypes']).all()):
            raise ValueError('existing text prototype mismatch')
    for split, rows in manifest['splits'].items():
        path = args.output_dir/f'{split}.pt'
        if path.exists():
            validate_existing_split(path, split, rows, manifest['manifest_sha256'])
    if not meta_path.exists():
        atomic_json_save(metadata, meta_path)
    pending = [k for k in ('train','val','test') if not (args.output_dir/f'{k}.pt').exists()]
    if prototype_path.exists() and not pending:
        return
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    from transformers import CLIPModel, CLIPTokenizerFast
    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(str(args.model_snapshot), local_files_only=True, use_safetensors=False).eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    if not prototype_path.exists():
        tokenizer = CLIPTokenizerFast.from_pretrained(str(args.model_snapshot), local_files_only=True)
        prototypes, scale = encode_text(model,tokenizer,names,device)
        atomic_torch_save({'text_prototypes':prototypes, 'logit_scale':scale, 'class_names':names,
            'prompt_template':PROMPT, 'model_weight_sha256':weight_hash, 'manifest_sha256':manifest['manifest_sha256']},prototype_path)
        print('saved',prototype_path,flush=True)
    for split in pending:
        rows = manifest['splits'][split]
        dataset = ManifestImageDataset(rows,image_root,transform)
        features = encode_images(model,dataset,device,args.batch_size,args.workers,split)
        atomic_torch_save({'features':features,'targets':torch.tensor([row['label'] for row in rows]),
            'manifest_sha256':manifest['manifest_sha256'],'split':split},args.output_dir/f'{split}.pt')
        print('saved',args.output_dir/f'{split}.pt',flush=True)


if __name__ == '__main__':
    main()
