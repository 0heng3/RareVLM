"""Apply corrected admission boundaries to the same original training-only inputs."""

import argparse
from pathlib import Path
import json

from prepare_semantic_admission_v1 import digest, sha, write_once
from vlm_semantic_admission_v2 import POLICY, build_training_view


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUTS / 'vlm_semantic_admission_v2')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path('/opt/data/private')):
        raise ValueError('Output must remain persistent and private')
    previous = OUTPUTS / 'vlm_semantic_admission_v1'
    receipt = json.loads((previous / 'PREPARATION_RESULT.json').read_text())
    paths = {name: Path(receipt['inputs'][name]['path']) for name in
             ('source_train_only_export', 'pretraining_review')}
    for name, path in paths.items():
        if sha(path) != receipt['inputs'][name]['sha256']:
            raise ValueError('Original admission source changed: ' + name)
    write_once(args.output / 'ADMISSION_POLICY.json', POLICY)
    source = json.loads(paths['source_train_only_export'].read_text())
    if source.get('manifest_sha256') != digest({k: v for k, v in source.items() if k != 'manifest_sha256'}):
        raise ValueError('Original training manifest identity mismatch')
    prior = json.loads(paths['pretraining_review'].read_text())
    additional_path = previous / 'ADDITIONAL_TRAINING_REVIEW.json'
    additional = json.loads(additional_path.read_text())
    view, decisions, qa, summary = build_training_view(source, prior, additional)
    view['manifest_sha256'] = digest(view)
    baseline = json.loads((previous / 'TRAINING_MANIFEST.json').read_text())
    unchanged = view['splits'] == baseline['splits'] and view['images'] == baseline['images']
    summary.update(status='completed', actual_rows_and_images_identical_to_v1=unchanged,
                   literal_backslash_n_prompts=sum(row.get('prompt_suffix', '').startswith('\\n')
                                                   for row in view['splits']['train_all']),
                   prompt_first_codepoints=sorted({ord(row['prompt_suffix'][0])
                                                   for row in view['splits']['train_all']
                                                   if row.get('prompt_suffix')}))
    files = list(paths.values()) + [additional_path, previous / 'TRAINING_MANIFEST.json',
             Path(__file__), PACKAGE / 'src/vlm_semantic_admission_v2/admission.py',
             PACKAGE / 'src/vlm_semantic_admission_v1/admission.py',
             PACKAGE / 'tools/prepare_semantic_admission_v1.py']
    summary['sources'] = [{'path': str(path), 'sha256': sha(path)} for path in files]
    for filename, value in [('TRAINING_MANIFEST.json', view), ('PAIR_ADMISSION.json', decisions),
                            ('QA_ADMISSION.json', qa), ('PREPARATION_RESULT.json', summary)]:
        write_once(args.output / filename, value)
    print(json.dumps({k: v for k, v in summary.items() if k != 'sources'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
