"""A stronger full-image teacher vs the current SFT student; no optimizer work."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import time

from rarevlm_paths import PROJECT_ROOT
from vlm_deep.common import collate, content_sha, encode, sha_file, write_json
from vlm_grpo_iteration_v1.common import chosen_sample, read
from prepare_stvqa_source_subset_v1 import literal

BASE = PROJECT_ROOT.parents[1] / 'outputs'
ROOT = BASE / 'vlm_full_teacher_signal_v1'
ASSETS = BASE / 'vlm_full_teacher_assets_v1'
PARENT = BASE / 'vlm_stvqa_replay_v1'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_FULL_TEACHER_SIGNAL_V1_PROTOCOL.md'
ROLES = ('student', 'teacher')


def prepare():
    import torch
    from vlm_probe.evaluate import inference_row, processor_for
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Teacher signal study already frozen')
    torch.set_num_threads(2)
    config = read(PARENT / 'CONFIG.json'); base = read(config['baseline_config'])
    original = read(PARENT / 'data/MANIFEST.json'); rows = original['evaluations']['native_hard']
    if len(rows) != 128 or len({q['image_id'] for q in rows}) != 128:
        raise ValueError('Require original128 distinct native training scenes')
    train_ids = {q['question_id'] for q in original['rows']}
    if not {q['question_id'] for q in rows} <= train_ids:
        raise ValueError('Teacher probe is restricted to the existing training pool')
    processors = {'student': processor_for(Path(base['model_root']), read(base['reference_config']), 1024),
                  'teacher': processor_for(ASSETS / 'model', read(base['reference_config']), 1024)}
    vocabulary = processors['student'].tokenizer.get_vocab()
    if processors['teacher'].tokenizer.get_vocab() != vocabulary:
        raise ValueError('Teacher and student actual token-to-ID mappings differ')
    special = {role: {name: proc.tokenizer.convert_tokens_to_ids(name) for name in
        ('<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>')}
        for role, proc in processors.items()}
    if special['student'] != special['teacher']:
        raise ValueError('Special token identity differs')
    valid = sorted(set(vocabulary.values()))
    sizes = {'student': read(Path(base['model_root']) / 'config.json')['vocab_size'],
             'teacher': read(ASSETS / 'model/config.json')['vocab_size']}
    if max(valid) >= min(sizes.values()):
        raise ValueError('Actual tokenizer ID is outside a model output head')
    images = {q['image_id']: original['images'][q['image_id']] for q in rows}; cpu = {}
    for index, q in enumerate(rows):
        encodings = {}
        for role, processor in processors.items():
            sample = chosen_sample(processor, q, images)
            prefix = encode(processor, inference_row(q), Path(images[q['image_id']]['path']), False)
            if (sample['input_ids'].shape[1] > 2048 or content_sha(prefix['input_ids'].tolist()) != q['inference_input_sha256']
                or prefix['image_grid_thw'][0].tolist() != q['expected_grid']):
                raise ValueError('Fixed teacher input frame or budget differs')
            encodings[role] = {'input_sha256': content_sha(sample['input_ids'].tolist()),
                'labels_sha256': content_sha(sample['labels'].tolist()),
                'pixels_sha256': hashlib.sha256(sample['pixel_values'].contiguous().numpy().tobytes()).hexdigest(),
                'grid': sample['image_grid_thw'].tolist()}
        if encodings['student'] != encodings['teacher']:
            raise ValueError('Actual multimodal/GT inputs differ: ' + q['question_id'])
        cpu[q['question_id']] = encodings['student']
        if (index + 1) % 16 == 0:
            write_json(ROOT / 'CPU_PROGRESS.json', {'completed': index + 1, 'expected': 128, 'gpu_calls': 0})
    data = {'rows': rows, 'images': images}
    write_json(ROOT / 'DATA.json', data); write_json(ROOT / 'CPU_INPUTS.json', cpu)
    write_json(ROOT / 'VALID_IDS.json', valid)
    baseline = read(PARENT / 'runs/ce/internal/native_hard/PREDICTIONS.json')
    if [(q['question_id'], q['inference_input_sha256']) for q in rows] != [(q['question_id'], q['input_sha256']) for q in baseline]:
        raise ValueError('Reused current-student predictions differ')
    write_json(ROOT / 'student_predictions.json', baseline)
    trained = read(PARENT / 'runs/ce/train/TRAIN_COMPLETE.json')
    source = read(ASSETS / 'MODEL_SOURCE.json')
    paths = [Path(__file__), PROTOCOL, ROOT / 'DATA.json', ROOT / 'CPU_INPUTS.json', ROOT / 'VALID_IDS.json',
        ROOT / 'student_predictions.json', ASSETS / 'MODEL_SOURCE.json', PARENT / 'runs/ce/train/TRAIN_COMPLETE.json',
        Path(config['baseline_config']), Path(base['reference_config']),
        PROJECT_ROOT / 'tools/run_stvqa_headroom_source_adapter_v1.py']
    paths += [Path(trained['adapter']) / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
    paths += [p for p in (ASSETS / 'model').iterdir() if p.is_file() and p.suffix != '.safetensors' and not p.name.endswith(('.partial', '.prefetch'))]
    paths += [Path(meta['path']) for meta in images.values()]
    paths += [PROJECT_ROOT / 'src' / p for p in ('vlm_native_scope_v1/model.py', 'vlm_batch_stable_v1/model.py',
        'vlm_fp32_readout_v1/model.py', 'vlm_fit_positive_control_v1/study.py', 'research_runtime/queue.py')]
    write_json(ROOT / 'PLAN.json', {'status': 'frozen', 'files': {str(p): sha_file(p) for p in paths},
        'teacher_repository': source['repository'], 'teacher_revision': source['revision'],
        'teacher_weights': [f for f in source['files'] if f['name'].endswith('.safetensors')],
        'student_adapter': trained['adapter'], 'student_lora_sha256': trained['final_lora_sha256'],
        'model_root': base['model_root'], 'reference_config': base['reference_config'],
        'qa': 128, 'actual_vocabulary': len(valid), 'head_vocab_sizes': sizes, 'special_ids': special['student'],
        'new_generations': 128, 'teacher_forced_sequences': 256, 'optimizer_updates': 0, 'automatic_training': False})
    print('CPU_TEACHER_INPUT_COMPATIBILITY_PASSED', {'qa': 128, 'actual_tokens': len(valid), 'head_sizes': sizes}, flush=True)


def verify():
    plan = read(ROOT / 'PLAN.json')
    for path, digest in plan['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Frozen teacher probe source/input changed: ' + path)
    assets = read(ASSETS / 'TRANSPORT_COMPLETE.json')
    if not assets['all_pinned_official_hashes_matched'] or assets['revision'] != plan['teacher_revision']:
        raise ValueError('Teacher asset verification incomplete')
    for item in plan['teacher_weights']:
        if (ASSETS / 'model' / item['name']).stat().st_size != item['size']:
            raise ValueError('Verified teacher weight size changed')
    return plan


def worker(role):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration
    from vlm_native_scope_v1.model import load
    from vlm_fp32_readout_v1.model import install_fp32_readout
    from vlm_batch_stable_v1.model import install
    from vlm_human_sft.evaluate import lora_hash
    from vlm_fit_positive_control_v1.study import token_statistics
    from vlm_probe.evaluate import processor_for
    from vlm_sft.common import to_device
    from run_stvqa_headroom_source_adapter_v1 import source_generated
    if role not in ROLES:
        raise ValueError('Unknown teacher-probe role')
    plan = verify(); data = read(ROOT / 'DATA.json'); cpu = read(ROOT / 'CPU_INPUTS.json')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); directory = ROOT / 'runs' / role; directory.mkdir(parents=True, exist_ok=True)
    if role == 'student':
        model, processor, binding = load(plan['student_adapter'], 'visual', False)
        if lora_hash(model) != plan['student_lora_sha256']:
            raise ValueError('Wrong frozen student')
        predictions = {q['question_id']: q for q in read(ROOT / 'student_predictions.json')}
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ASSETS / 'model'), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True, device_map={'': 'cuda:0'})
        model.eval().requires_grad_(False)
        binding = {'readout': install_fp32_readout(model),
                   'numerics': install(SimpleNamespace(get_base_model=lambda: model), 'imagewise_fp32_language')}
        processor = processor_for(ASSETS / 'model', read(plan['reference_config']), 1024)
        predictions = {}
    model.eval().requires_grad_(False); eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    valid = torch.tensor(read(ROOT / 'VALID_IDS.json'), device='cuda:0'); completed = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(data['rows']), 4):
            rows = data['rows'][start:start + 4]
            if role == 'teacher':
                predictions.update({q['question_id']: q for q in source_generated(model, processor, rows, data)})
            samples = [chosen_sample(processor, q, data['images']) for q in rows]
            for q, sample in zip(rows, samples):
                if content_sha(sample['input_ids'].tolist()) != cpu[q['question_id']]['input_sha256'] or content_sha(sample['labels'].tolist()) != cpu[q['question_id']]['labels_sha256']:
                    raise ValueError('Actual teacher GT prefix changed')
            batch = to_device(collate(samples, processor.tokenizer.pad_token_id), torch.device('cuda:0'))
            keep = int(batch['labels'].ne(-100).sum(1).max()) + 1
            hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -keep:, :],))
            try:
                output = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
            finally:
                hook.remove()
            stats = token_statistics(output.logits, batch['labels'], eos)
            targets = batch['labels'][:, -output.logits.shape[1]:][:, 1:]
            for index, (q, stat) in enumerate(zip(rows, stats)):
                body = targets[index].ne(-100) & targets[index].ne(eos)
                full = output.logits[index, :-1][body].float(); shared = full.index_select(-1, valid)
                padding = {str(t): float((1 - (torch.logsumexp(shared / t, -1) - torch.logsumexp(full / t, -1)).exp()).clamp_min(0).max()) for t in (1, 2)}
                labels = targets[index][body]
                path = directory / 'logits' / (q['question_id'] + '.pt'); path.parent.mkdir(exist_ok=True)
                torch.save({'logits': shared.cpu(), 'labels': labels.cpu()}, path)
                pred = predictions[q['question_id']]
                if pred['input_sha256'] != q['inference_input_sha256']:
                    raise ValueError('Actual teacher generation input changed')
                completed.append({'question_id': q['question_id'], 'image_id': q['image_id'], 'prediction': pred['prediction'],
                    'terminated': pred['terminated'], 'literal_correct': bool(pred['terminated'] and literal(pred['prediction']) == literal(q['target'])),
                    'body_nll': stat['content_ce'], 'eos_nll': stat['eos_ce'], 'body_all_argmax': stat['content_target_argmax'],
                    'padding_mass_max': padding, 'logits_sha256': sha_file(path)})
            del output, batch, samples
            progress = {'status': 'frozen_teacher_signal', 'completed': len(completed), 'expected': 128,
                'new_generations': len(completed) if role == 'teacher' else 0, 'teacher_forced_sequences': len(completed),
                'optimizer_updates': 0, 'active_seconds': time.monotonic() - started, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            write_json(directory / 'ROWS.json', completed); write_json(directory / 'PROGRESS.json', progress)
            if progress['active_seconds'] > 900 or progress['peak_allocated_gib'] > 65:
                raise RuntimeError('Teacher-probe budget exceeded')
    if role == 'student' and lora_hash(model) != plan['student_lora_sha256']:
        raise ValueError('Student LoRA changed during zero-update probe')
    write_json(directory / 'COMPLETE.json', {**progress, 'status': 'completed', 'binding': binding})


def analyze():
    import torch
    import torch.nn.functional as F
    plan = verify(); data = read(ROOT / 'DATA.json'); valid = torch.tensor(read(ROOT / 'VALID_IDS.json'))
    lookup = {int(v): i for i, v in enumerate(valid.tolist())}
    records = {role: read(ROOT / 'runs' / role / 'ROWS.json') for role in ROLES}
    if any([q['question_id'] for q in rows] != [q['question_id'] for q in data['rows']] for rows in records.values()):
        raise ValueError('Incomplete teacher signal rows')
    diagnostics = []; corrections = []; regressions = 0; informative = 0
    for row, student, teacher in zip(data['rows'], records['student'], records['teacher']):
        qid = row['question_id']; tensors = {}
        for role, record in [('student', student), ('teacher', teacher)]:
            path = ROOT / 'runs' / role / 'logits' / (qid + '.pt')
            if sha_file(path) != record['logits_sha256']:
                raise ValueError('Saved soft targets changed')
            tensors[role] = torch.load(path, map_location='cpu', weights_only=True)
        s, t = tensors['student'], tensors['teacher']
        if not torch.equal(s['labels'], t['labels']) or s['logits'].shape != t['logits'].shape:
            raise ValueError('Teacher/student soft target frame differs')
        labels = torch.tensor([lookup[int(i)] for i in s['labels']]); measures = {}
        for temperature in (1, 2):
            ps = (s['logits'] / temperature).softmax(-1); pt = (t['logits'] / temperature).softmax(-1)
            gt = pt.gather(1, labels[:, None]); uniform = ((1 - gt) / (pt.shape[1] - 1)).expand_as(pt).clone()
            uniform.scatter_(1, labels[:, None], gt)
            real = temperature * (ps - pt); null = temperature * (ps - uniform)
            measures[str(temperature)] = {'cos_to_same_gt_mass_uniform': float(F.cosine_similarity(real.flatten(), null.flatten(), dim=0)),
                'relative_gradient_difference': float((real - null).norm() / real.norm().clamp_min(1e-12)),
                'teacher_non_gt_mass_mean': float((1 - gt).mean())}
        novel = not student['literal_correct'] and teacher['literal_correct']
        if novel:
            corrections.append(student['body_nll'] - teacher['body_nll'])
        regressions += bool(student['literal_correct'] and not teacher['literal_correct'])
        eligible = teacher['literal_correct'] and teacher['body_nll'] <= 1 and teacher['eos_nll'] <= .5
        information = eligible and measures['2']['cos_to_same_gt_mass_uniform'] < .995 and measures['2']['relative_gradient_difference'] >= .1
        informative += bool(information)
        diagnostics.append({'question_id': qid, 'student': student, 'teacher': teacher, 'temperatures': measures,
                            'novel_teacher_correction': novel, 'informative_at_fixed_T2': bool(information)})
    correct = sum(q['literal_correct'] for q in records['teacher'])
    unique = len({literal(row['target']) for row, teacher in zip(data['rows'], records['teacher']) if teacher['literal_correct']})
    drop = sum(corrections) / len(corrections) if corrections else None
    padding = max(q['padding_mass_max']['2'] for rows in records.values() for q in rows)
    gates = {'teacher_correct48_distinct40': correct >= 48 and unique >= 40,
             'novel16_net8': len(corrections) >= 16 and len(corrections) - regressions >= 8,
             'novel_mean_nll_drop_point2': drop is not None and drop >= .2,
             'non_gt_structured_information16': informative >= 16, 'padding_mass_at_T2_at_most_point01': padding <= .01}
    result = {'status': 'completed', 'teacher_literal_correct': correct, 'teacher_correct_distinct_targets': unique,
        'student_literal_correct': sum(q['literal_correct'] for q in records['student']), 'novel_corrections': len(corrections),
        'regressions': regressions, 'novel_mean_nll_drop': drop, 'informative_at_T2': informative,
        'max_padding_mass_T2': padding, 'gates': gates, 'teacher_signal': all(gates.values()),
        'new_generations': 128, 'teacher_forced_sequences': 256, 'optimizer_updates': 0, 'automatic_training': False,
        'goal_complete': False, 'scope': 'Fixed known128 training questions; teacher admission only, not student ability gain.'}
    write_json(ROOT / 'DIAGNOSTICS.json', diagnostics); write_json(ROOT / 'ANALYSIS.json', result)
    write_json(ROOT / 'DECISION.json', {'status': 'completed', 'decision': 'design_separate_matched_KD_student_protocol' if result['teacher_signal'] else 'stop_this_fixed_full_teacher_configuration', 'automatic_training': False})
    write_json(ROOT / 'COMPLETION.json', result); print('FULL_TEACHER_SIGNAL_COMPLETE', result, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', choices=ROLES); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.worker:
        worker(args.worker); return
    registration = read(ROOT / 'REGISTRATION.json')
    for path, digest in registration['files'].items():
        if sha_file(path) != digest:
            raise ValueError('Preregistered teacher probe source/protocol changed: ' + path)
    waiting_started = time.monotonic()
    while not (ASSETS / 'TRANSPORT_COMPLETE.json').exists():
        if time.monotonic() - waiting_started > 2100:
            raise TimeoutError('Verified teacher assets did not become ready within the acquisition window')
        progress = read(ASSETS / 'TRANSPORT_PROGRESS.json') if (ASSETS / 'TRANSPORT_PROGRESS.json').exists() else {}
        if progress.get('status') == 'failed':
            raise RuntimeError('Teacher asset acquisition failed; no GPU admitted')
        write_json(ROOT / 'STATUS.json', {'status': 'waiting_verified_teacher_assets', 'optimizer_updates': 0})
        time.sleep(10)
    if not (ROOT / 'PLAN.json').exists():
        write_json(ROOT / 'STATUS.json', {'status': 'cpu_input_compatibility', 'optimizer_updates': 0})
        prepare()
    verify()
    from research_runtime.queue import run_queue
    entry = str(Path(__file__).resolve())
    write_json(ROOT / 'STATUS.json', {'status': 'gpu_teacher_diagnostic', 'optimizer_updates': 0})
    run_queue(ROOT / 'queue', [{'key': role, 'command': [sys.executable, '-u', entry, '--worker', role],
        'completion_marker': str(ROOT / 'runs' / role / 'COMPLETE.json')} for role in ROLES], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    write_json(ROOT / 'STATUS.json', {'status': 'analyzing_teacher_information', 'optimizer_updates': 0})
    analyze()
    write_json(ROOT / 'STATUS.json', {'status': 'completed', 'optimizer_updates': 0})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        phase = read(ROOT / 'STATUS.json').get('status') if (ROOT / 'STATUS.json').exists() else 'startup'
        write_json(ROOT / 'RUNTIME_ERROR.json', {'status': 'failed', 'phase': phase,
            'error_type': type(error).__name__, 'message': str(error), 'optimizer_updates': 0})
        raise
