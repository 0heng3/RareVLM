"""Check real saved displacements and CPU-shadow Adam directions; no VLM loading."""

from pathlib import Path
import hashlib
import json
import time

import torch
from safetensors.torch import load_file

from vlm_gradient_mechanism_v1.adam import flatten_parameters, moment_delta, projection, shadow_delta


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE.parents[1] / 'outputs'
ROOT = OUTPUTS / 'vlm_adam_direction_v1'
GRAD = OUTPUTS / 'vlm_gradient_mechanism_v1'
TRAIN = OUTPUTS / 'vlm_complementary_rank_v1/runs/ce/train'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parameter_hash(named):
    digest = hashlib.sha256()
    for name, value in named.items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    cohort = read(GRAD / 'COHORT.json')
    checkpoints = {}
    paths = [GRAD / 'COHORT.json']
    for step in (1, 175, 200):
        options = list(TRAIN.glob(f'checkpoint_{step:06d}_*/state.pt'))
        if len(options) != 1:
            raise ValueError(f'Expected one immutable checkpoint at {step}')
        paths.append(options[0])
        checkpoints[step] = torch.load(options[0], map_location='cpu', weights_only=False)
        if checkpoints[step]['completed_updates'] != step:
            raise ValueError('Checkpoint cursor differs')
    final = checkpoints[200]
    names = list(final['trainable_parameters'])
    for state in checkpoints.values():
        if (list(state['trainable_parameters']) != names
                or state['lineage_sha256'] != final['lineage_sha256']
                or state['schedule_sha256'] != final['schedule_sha256']):
            raise ValueError('Saved optimizer coordinate lineage differs')
        if parameter_hash(state['trainable_parameters']) != state['trainable_tensor_sha256']:
            raise ValueError('Checkpoint parameter identity differs')
    if final['trainable_tensor_sha256'] != cohort['models']['ce']['lora_sha256']:
        raise ValueError('Final gradient and checkpoint identities differ')
    initial_path = Path(cohort['models']['initial']['adapter_weights']['path'])
    paths.append(initial_path)
    if sha(initial_path) != cohort['models']['initial']['adapter_weights']['sha256']:
        raise ValueError('Initial adapter changed')
    initial_weights = load_file(str(initial_path))
    initial_named = {name: initial_weights[name.replace('.default.', '.')] for name in names}
    if parameter_hash(initial_named) != cohort['models']['initial']['lora_sha256']:
        raise ValueError('Reconstructed initial parameter order/hash differs')
    initial_flat = torch.cat([v.double().reshape(-1) for v in initial_named.values()])
    gradients = {}
    for role in ('initial', 'ce'):
        vector_path = GRAD / 'runs' / role / 'gradient_vectors.pt'
        summary_path = GRAD / 'runs' / role / 'ANALYSIS.json'
        paths.extend([vector_path, summary_path])
        gradients[role] = torch.load(vector_path, map_location='cpu', weights_only=True)
        recorded = read(summary_path)['standard_ce_direction']
        for key, value in gradients[role].items():
            if value.numel() != initial_flat.numel() or value.dtype != torch.float64:
                raise ValueError('Saved gradient dimensions/precision differ')
            expected = recorded['groups'][key]['norm']
            if abs(float(value.norm()) - expected) > max(1e-10, expected * 1e-10):
                raise ValueError('Saved gradient norm differs from original result')
    schedule_path = OUTPUTS / 'vlm_complementary_rank_v1/data/SCHEDULE.json'
    paths.append(schedule_path)
    first = read(schedule_path)['updates'][0]['micros']
    first_qids = {source: [q for micro in first if micro['source'] == source for q in micro['question_ids']]
                  for source in ('human', 'text')}
    same_first_batch = (first_qids['human'] == cohort['groups']['ordinary_human']
                        and first_qids['text'] == cohort['groups']['text'])
    coord = []
    offset = 0
    for name, p in final['trainable_parameters'].items():
        coord.append({'name': name, 'shape': list(p.shape), 'start': offset, 'stop': offset + p.numel()})
        offset += p.numel()
    source_paths = paths + [Path(__file__), PACKAGE / 'src/vlm_gradient_mechanism_v1/adam.py',
                           PACKAGE / 'src/vlm_object_train/train.py',
                           PACKAGE / 'src/vlm_complementary_rank/train.py',
                           PACKAGE / 'tools/run_vlm_gradient_mechanism_v1.py']
    source_hashes = {str(path): sha(path) for path in source_paths}
    write(ROOT / 'PARAMETER_ORDER_RECONSTRUCTED.json', {
        'coordinates': coord, 'gradient_names_independently_saved_in_v1': False,
        'basis': 'unchanged named_parameters traversal in both producers and ordered saved trainable_parameters; hashes/shapes checked',
        'not_a_new_live_gpu_coordinate_check': True})
    measures = {}
    probes = ('probe_no_answer', 'probe_yes_answer')

    def record(name, role, delta, scope):
        measures[name] = {'gradient_checkpoint': role, 'scope': scope,
                          'probes': {q: projection(gradients[role][q], delta) for q in probes}}

    first_actual = flatten_parameters(checkpoints[1]) - initial_flat
    record('actual_first_parameter_displacement', 'initial', first_actual,
           'Observed FP32 theta1-theta0; probe gradient at theta0. Local first-order estimate, not remeasured loss.')
    record('first_step_moment_reconstruction', 'initial', moment_delta(checkpoints[1]),
           'FP64 ideal increment from post-step1 Adam moments; excludes FP32 rounding/kernel differences.')
    record('last25_observed_displacement_at_final_gradient', 'ce',
           flatten_parameters(final) - flatten_parameters(checkpoints[175]),
           'Observed theta200-theta175 with endpoint theta200 gradient; not an integrated loss change over 25 steps.')
    record('last_step200_moment_direction_at_final_gradient', 'ce', moment_delta(final),
           'Ideal step200 moment-derived increment evaluated with theta200 gradient; theta199 gradient unavailable.')
    shadow_receipts = {}
    for role in ('initial', 'ce'):
        g = gradients[role]
        standard = .5 * g['human_ce'] + .5 * g['text_ce']
        answer3 = standard + (1 / 3) * g['human_answer']
        scale = checkpoints[1]['optimizer']['param_groups'][0]['lr'] if role == 'initial' else final['optimizer']['param_groups'][0]['lr']
        record(f'{role}_raw_sgd_at_recorded_lr', role, -scale * standard,
               'Hypothetical unpreconditioned SGD comparison using the same LR as the shadow.')
        if role == 'initial':
            copied_state = {'trainable_parameters': initial_named,
                            'optimizer': {'state': {}, 'param_groups': checkpoints[1]['optimizer']['param_groups']}}
        else:
            copied_state = final
        for condition, vector in [('standard', standard), ('answer3', answer3)]:
            delta, receipt = shadow_delta(copied_state, vector, clip_norm=1.)
            name = f'{role}_cpu_shadow_adam_{condition}'
            record(name, role, delta,
                   'CPU installed AdamW on cloned FP32 tensors/moments, not a live-model update. Final uses last saved LR for a counterfactual next step, not a resumed schedule.')
            shadow_receipts[name] = receipt
    one = checkpoints[1]['optimizer']
    first_applied = torch.cat([(one['state'][pid]['exp_avg'].double() / (1 - one['param_groups'][0]['betas'][0])).reshape(-1)
                               for pid in one['param_groups'][0]['params']])
    initial_standard = .5 * gradients['initial']['human_ce'] + .5 * gradients['initial']['text_ce']
    result = {'status': 'completed', 'model_loads': 0, 'gpu_forwards': 0, 'trained_model_updates': 0,
              'copied_cpu_optimizer_steps': sum(v['copied_cpu_optimizer_steps'] for v in shadow_receipts.values()),
              'same_first_batch_as_gradient_probe_sources': same_first_batch,
              'first_applied_gradient_vs_saved_probe_objective': {
                  'relative_l2_difference': float((first_applied - initial_standard).norm() / first_applied.norm()),
                  'cosine': float(torch.dot(first_applied, initial_standard) / first_applied.norm() / initial_standard.norm())},
              'first_moment_increment_vs_observed_displacement_relative_l2':
                  float((moment_delta(checkpoints[1]) - first_actual).norm() / first_actual.norm()),
              'measurements': measures, 'cpu_shadow_receipts': shadow_receipts,
              'saved_probe_first_answer_nll': {role: read(GRAD / 'runs' / role / 'ANALYSIS.json')['actual_batches'][0]['first_answer_ce']
                                                for role in ('initial', 'ce')},
              'parameter_coordinate_scope': 'v1 did not save names alongside gradient vectors; reconstructed from immutable producer order and ordered checkpoint tensors, not an independent live check',
              'torch_version': torch.__version__, 'cpu_seconds': time.monotonic() - started,
              'limits': ['Four training probes, not all errors or heldout capability.',
                         'CPU shadow and FP64 moment math are not bit-identical historical GPU foreach execution.',
                         'Endpoint linearization of a long displacement is not historical cumulative loss causality.']}
    if any(sha(path) != before for path, before in source_hashes.items()):
        raise ValueError('Read-only source changed during CPU analysis')
    result['sources_unchanged'] = True
    result['source_sha256'] = source_hashes
    write(ROOT / 'ANALYSIS.json', result)
    write(ROOT / 'COMPLETION.json', {'status': 'completed', 'trained_model_updates': 0,
          'copied_cpu_optimizer_steps': result['copied_cpu_optimizer_steps'], 'gpu_forwards': 0,
          'cpu_seconds': result['cpu_seconds'], 'analysis_path': str(ROOT / 'ANALYSIS.json')})
    print(json.dumps({name: {q: v['first_order_loss_change'] for q, v in item['probes'].items()}
                      for name, item in measures.items()}, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k in ('cpu_seconds', 'same_first_batch_as_gradient_probe_sources',
          'first_applied_gradient_vs_saved_probe_objective', 'first_moment_increment_vs_observed_displacement_relative_l2')}, indent=2))


if __name__ == '__main__':
    main()
