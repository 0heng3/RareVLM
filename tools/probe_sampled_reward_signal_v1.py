"""Frozen32 x8 original-policy samples: preserve actions, measure reward support, zero updates."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

from rarevlm_paths import PROJECT_ROOT, SOURCE_ROOT
from vlm_deep.common import collate, content_sha, encode, sha_file, write_json
from vlm_sampled_signal_v1.trajectory import attach_completion, group_signal, stop_at_eos

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_sampled_reward_signal_v1'
TRACE = ROOT.parent / 'vlm_error_decision_trace_v1'
NUMERIC = ROOT.parent / 'vlm_error_numeric_path_v1'
OLD = ROOT.parent / 'vlm_error_supervision_sft_v1'
UPSTREAM = ROOT.parent / 'vlm_frontier_bottleneck_review_20261019/upstream/huggingface__trl'
PROTOCOL = PROJECT_ROOT / 'docs/VLM_SAMPLED_REWARD_SIGNAL_V1_PROTOCOL.md'
RESULT_DOC = PROJECT_ROOT / 'docs/VLM_SAMPLED_REWARD_SIGNAL_V1_RESULTS.md'
PAGE = PROJECT_ROOT / 'docs/learning/同题采样与GRPO信号_真实token轨迹.html'


def read(path):
    return json.loads(Path(path).read_text())


def link(path):
    return quote(os.path.relpath(path, PAGE.parent), safe='/._-')


def processor_for_plan(plan):
    from vlm_probe.evaluate import processor_for
    return processor_for(Path(plan['model_root']), read(plan['reference_config']), 1024)


def render(result=None):
    data = read(ROOT / 'INPUTS.json')
    records = read(ROOT / 'CPU_VERIFICATION.json')['cases']
    cards = []
    for role, qid in [('initial', '4604'), ('initial', '9964'), ('dpo_sft', '9964')]:
        q = next(q for q in data['rows'] if q['question_id'] == qid)
        record = next(r for r in records if r['role'] == role and r['question_id'] == qid)
        cards.append(f'<article><h3>{html.escape(role)} · q{qid}</h3>'
            f'<img src="{link(data["images"][q["image_id"]]["path"])}" alt="原始训练图" loading="lazy">'
            f'<p>{html.escape(q["question"])}</p><p>目标：{html.escape(q["target"])}；实际生成：{html.escape(record["prediction"])}</p>'
            f'<p>实际动作 <code>{record["content_ids"]}</code>：{html.escape(str(record["content_tokens"]))}</p>'
            f'<p>文本重分词 <code>{record["canonical_ids"]}</code>：{html.escape(str(record["canonical_tokens"]))}</p>'
            '<p class="note">上面来自已完成的真实greedy记录；同一文本可以有不同BPE路径。图为原图展示，不是attention或新增采样。</p></article>')
    sampled = ''
    if (ROOT / 'GROUPS.json').exists():
        for group in read(ROOT / 'GROUPS.json'):
            q = next(q for q in data['rows'] if q['question_id'] == group['question_id'])
            sampled += f'<article><h3>q{q["question_id"]} · 清晰奖励对：{group["signal"]["clear_reward_pair"]}</h3>'
            sampled += f'<img src="{link(data["images"][q["image_id"]]["path"])}" loading="lazy" alt="原始训练图"><p>{html.escape(q["question"])}；原目标：{html.escape(q["target"])}</p>'
            sampled += '<table><tr><th>采样文本</th><th>VQA奖励</th><th>优势A</th><th>自然EOS</th></tr>'
            for answer, adv in zip(group['samples'], group['signal']['advantages']):
                sampled += f'<tr><td>{html.escape(answer["prediction"])}</td><td>{answer["reward"]:.3f}</td><td>{"删失" if adv is None else format(adv,".3f")}</td><td>{answer["terminated"]}</td></tr>'
            sampled += '</table></article>'
    status = '已准备并通过CPU验证，GPU采样尚未完成；0优化器更新。' if result is None else (
        f'采样完成：{result["sample_count"]}条；清晰正负组{result["clear_reward_pair_groups"]}/32，信号必要条件通过={result["signal_gate_passed"]}。0优化器更新，不能称模型提分。')
    PAGE.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>同题采样与GRPO信号</title><style>body{max-width:1140px;margin:30px auto;padding:0 18px;font:17px/1.65 system-ui;background:#f8fafc;color:#1e293b}article{padding:16px;background:white;border:1px solid #cbd5e1;margin:16px 0}img{width:100%;max-height:480px;object-fit:contain}code,pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e2e8f0;padding:3px}.note{background:#fff7ed;padding:12px}td,th{border:1px solid #cbd5e1;padding:7px}table{border-collapse:collapse;width:100%}</style>'
        '<h1>同一道题，8种实际动作：在线奖励学习先需要什么？</h1>'
        f'<p class="note">{status}</p><p>上一轮DPO剩余23错题中22题的正确整段概率仍低于固定负例；这说明相对margin改善尚不足以改变决策，不能据此直接保证换GRPO有效。</p>'
        '<pre>原图＋问题 → 原始策略采样8次 → 原VQA评分 r₁…r₈\nAᵢ = (rᵢ − 同题平均r) / (同题样本标准差 + 0.0001)\n全为0或全为1 → A全部为0，组内没有区分信号\n保留实际prompt_ids＋completion_ids → 后续损失沿实际采样路径计算</pre>'
        '<p>例如一组奖励[1,0,0,0,0,0,0,0]，均值0.125、样本标准差约0.3536；正确样本A约+2.474，错误约−0.353。这是公式演示，不是新模型结果。A仅表示更新偏好方向，不能代替独立能力提升。</p>'
        '<p>在线学习必须使用实际采出的token。解码后重分词可能变成另一条路径；EOS后补位也不是动作。新接口用最早监督位置裁输出头，保留首答预测位置；图像/问题前缀全部mask。</p>'
        '<h2>已保存真实轨迹：相同文字不等于相同动作</h2>'+''.join(cards)+
        '<h2>新同题采样（训练信号，非独立评测）</h2>'+(sampled or '<p>尚无GPU采样结果。</p>')+
        f'<p><a href="{link(PROTOCOL)}">固定协议</a> · <a href="{link(RESULT_DOC)}">结果与限制</a> · <a href="{link(SOURCE_ROOT / "vlm_sampled_signal_v1/trajectory.py")}">实际轨迹/损失接口</a></p></html>')


def prepare():
    from vlm_probe.evaluate import inference_row
    from vlm_object_train.train import package_lock
    import torch
    torch.set_num_threads(2)
    if (ROOT / 'PLAN.json').exists():
        raise FileExistsError('Already frozen; inspect existing workflow')
    trace = read(TRACE / 'PLAN.json')
    config_path = PROJECT_ROOT / 'configs/vlm_error_supervision_sft_v1.json'
    config = read(config_path)
    base = read(config['baseline_config'])
    plan = {**trace['roles']['initial'], 'model_root': base['model_root'], 'reference_config': base['reference_config'],
            'questions': 32, 'samples_per_question': 8, 'max_new_tokens': 32,
            'temperature': 1., 'top_p': 1., 'top_k': 0, 'seed': 20261020,
            'minimum_clear_groups': 16, 'worker_active_cap_seconds': 600, 'training_updates': 0}
    processor = processor_for_plan(plan)
    data = read(TRACE / 'DATA.json')
    generations = {role: read(TRACE / role / 'GENERATION.json') for role in ('initial', 'sft', 'dpo_sft')}
    eos = generations['initial'][0]['eos_ids']
    cpu = []
    for row in data['rows']:
        prefix = encode(processor, inference_row(row), Path(data['images'][row['image_id']]['path']), False)
        if content_sha(prefix['input_ids'].tolist()) != row['inference_input_sha256'] or prefix['image_grid_thw'][0].tolist() != row['expected_grid']:
            raise ValueError('Original inference prefix changed')
        for role, records in generations.items():
            record = next(q for q in records if q['question_id'] == row['question_id'])
            ids, ended = stop_at_eos(record['generated_ids'], eos, 32)
            attached = attach_completion(prefix, ids)
            if attached['input_ids'][0, :prefix['input_ids'].shape[1]].tolist() != prefix['input_ids'][0].tolist():
                raise ValueError('Sampled prompt changed')
            if attached['labels'][attached['labels'] != -100].tolist() != ids:
                raise ValueError('Sampled actions changed')
            text = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            if text != record['prediction']:
                raise ValueError('Stored text view does not match actual tokens')
            content = ids[:-1] if ended else ids
            canonical = processor.tokenizer.encode(text, add_special_tokens=False)
            cpu.append({'role': role, 'question_id': row['question_id'], 'prediction': text,
                        'terminated': ended, 'content_ids': content, 'canonical_ids': canonical,
                        'content_tokens': processor.tokenizer.convert_ids_to_tokens(content),
                        'canonical_tokens': processor.tokenizer.convert_ids_to_tokens(canonical),
                        'content_path_differs': content != canonical,
                        'preserved_actual_actions': True})
    write_json(ROOT / 'INPUTS.json', {'rows': data['rows'], 'images': data['images']})
    write_json(ROOT / 'CPU_VERIFICATION.json', {'status': 'passed', 'prefixes': 32, 'actual_trajectories': len(cpu),
        'preserved_actual_action_trajectories': sum(q['preserved_actual_actions'] for q in cpu),
        'canonical_path_differences': sum(q['content_path_differs'] for q in cpu), 'cases': cpu,
        'training_updates': 0, 'scope': 'CPU interface verification using saved real actions, no new generation or model update.'})
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT / 'INPUTS.json', ROOT / 'CPU_VERIFICATION.json',
             PROJECT_ROOT / 'tests/test_vlm_sampled_signal_v1.py', TRACE / 'PLAN.json', TRACE / 'DATA.json',
             config_path, Path(config['baseline_config']), Path(base['reference_config']),
             UPSTREAM / 'COMMIT.json', UPSTREAM / 'source/trl/trainer/grpo_trainer.py']
    paths += [TRACE / role / 'GENERATION.json' for role in generations]
    paths += list((SOURCE_ROOT / 'vlm_sampled_signal_v1').glob('*.py'))
    paths += [SOURCE_ROOT / f for f in ('vlm_deep/common.py', 'vlm_deep/evaluate.py', 'vlm_sft/common.py',
              'vlm_probe/evaluate.py', 'vlm_fp32_readout_v1/model.py', 'vlm_fp32_adapter/model.py',
              'vlm_runtime_metadata_v1/model.py', 'vlm_attention_scope_repair_v1/generation.py', 'research_runtime/queue.py')]
    paths += [Path(plan['adapter']) / f for f in ('adapter_config.json', 'adapter_model.safetensors')]
    plan['files'] = {str(p): sha_file(p) for p in paths}
    plan['images'] = {v['path']: v['sha256'] for v in data['images'].values()}
    plan['packages'] = package_lock()
    plan['base_weight_files'] = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                 for p in sorted(Path(base['model_root']).glob('*.safetensors'))}
    write_json(ROOT / 'PLAN.json', plan)
    RESULT_DOC.write_text('# 原始策略同题采样：准备完成，尚无GPU结果\n\n'
        '固定32题×8次=256次采样、0优化器更新。已用保存的96条真实轨迹验证token直接拼接，实际动作均保持。'
        '新接口处理实际EOS与长度删失，输出头按最早监督位置裁尾；未更改旧DPO科学源。\n\n'
        '先等待数值隔离完成，再按共享空闲规则准入；不抢占已有任务。事前要求至少16/32组出现自然EOS的满分与低分回答，'
        '这只是在线优化信号必要条件，不能作为能力提分或自动训练授权。\n\n'
        '[固定协议](VLM_SAMPLED_REWARD_SIGNAL_V1_PROTOCOL.md) · [真实原图/token学习页](learning/同题采样与GRPO信号_真实token轨迹.html)\n')
    render()
    print('CPU_VERIFIED', len(cpu), 'real trajectories; frozen sampling plan', flush=True)


def worker():
    import torch
    from vlm_fp32_readout_v1.model import load
    from vlm_probe.evaluate import inference_row
    from vlm_sft.common import to_device
    from vlm_human_sft.evaluate import lora_hash
    from vlm_deep.evaluate import score_prediction
    from vlm_object_train.train import package_lock
    from vlm_attention_scope_repair_v1.generation import raw_greedy_config
    plan = read(ROOT / 'PLAN.json')
    for path, digest in {**plan['files'], **plan['images']}.items():
        if sha_file(path) != digest:
            raise ValueError('Frozen sampled-signal input changed: ' + path)
    if package_lock() != plan['packages']:
        raise ValueError('Sampling package versions changed')
    for path, value in plan['base_weight_files'].items():
        stat = Path(path).stat()
        if value != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            raise ValueError('Base storage changed')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    model, binding = load(plan['model_root'], plan['adapter'], torch.device('cuda:0'), training=False)
    processor = processor_for_plan(plan)
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Sampling original policy differs')
    model.eval()
    policy = raw_greedy_config(model.generation_config, plan['max_new_tokens'])
    policy.do_sample = True
    policy.top_k = plan['top_k']
    policy.temperature = plan['temperature']
    policy.top_p = plan['top_p']
    model.generation_config = policy
    write_json(ROOT / 'GENERATION_POLICY.json', policy.to_dict())
    write_json(ROOT / 'MODEL_IDENTITY.json', {'lora_sha256': lora_hash(model), 'binding': binding})
    eos = [policy.eos_token_id] if isinstance(policy.eos_token_id, int) else list(policy.eos_token_id)
    data = read(ROOT / 'INPUTS.json')
    groups = read(ROOT / 'GROUPS.json') if (ROOT / 'GROUPS.json').exists() else []
    if [g['question_id'] for g in groups] != [q['question_id'] for q in data['rows'][:len(groups)]]:
        raise ValueError('Sample-group resume order changed')
    torch.cuda.reset_peak_memory_stats()
    handle = model.get_output_embeddings().register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
    try:
        with torch.inference_mode():
            for row in data['rows'][len(groups):]:
                prefix = encode(processor, inference_row(row), Path(data['images'][row['image_id']]['path']), False)
                if content_sha(prefix['input_ids'].tolist()) != row['inference_input_sha256']:
                    raise ValueError('Sampling prompt is not original inference prompt')
                seed = (plan['seed'] + int(content_sha(row['question_id'])[:8], 16)) % (2**31)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                batch = to_device(collate([prefix] * plan['samples_per_question'], processor.tokenizer.pad_token_id), next(model.parameters()).device)
                out = model.generate(**batch, max_new_tokens=plan['max_new_tokens'], do_sample=True, use_cache=True,
                                     return_dict_in_generate=True, output_scores=True)
                start = batch['input_ids'].shape[1]
                generated = out.sequences[:, start:].tolist()
                selected_logps = torch.stack([score.float().log_softmax(-1).gather(
                    -1, out.sequences[:, start + j].unsqueeze(-1)).squeeze(-1)
                    for j, score in enumerate(out.scores)], dim=1).cpu().tolist()
                samples = []
                for i, raw in enumerate(generated):
                    ids, terminated = stop_at_eos(raw, eos, plan['max_new_tokens'])
                    attached = attach_completion(prefix, ids)
                    if attached['labels'][attached['labels'] != -100].tolist() != ids:
                        raise ValueError('Action IDs changed in learning view')
                    text = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                    reward = score_prediction(text, row, 'textvqa')
                    content = ids[:-1] if terminated else ids
                    canonical = processor.tokenizer.encode(text, add_special_tokens=False)
                    samples.append({'sample_index': i, 'completion_ids': ids, 'terminated': terminated,
                        'stop_reason': 'eos' if terminated else 'length_censored', 'prediction': text, 'reward': reward,
                        'sampling_token_logps': selected_logps[i][:len(ids)],
                        'canonical_content_ids': canonical, 'canonical_path_differs': content != canonical})
                signal = group_signal([q['reward'] for q in samples], [q['terminated'] for q in samples])
                groups.append({'question_id': row['question_id'], 'image_id': row['image_id'], 'seed': seed,
                    'prompt_ids': prefix['input_ids'][0].tolist(), 'image_grid_thw': prefix['image_grid_thw'].tolist(),
                    'samples': samples, 'signal': signal})
                write_json(ROOT / 'GROUPS.json', groups)
                allocated = torch.cuda.max_memory_allocated() / 2**30
                write_json(ROOT / 'PROGRESS.json', {'status': 'sampling', 'completed_groups': len(groups),
                    'expected_groups': 32, 'generated_samples': 8 * len(groups), 'active_seconds': time.monotonic() - started,
                    'peak_allocated_gib': allocated, 'training_updates': 0})
                if allocated > 65 or time.monotonic() - started > plan['worker_active_cap_seconds']:
                    raise RuntimeError('Registered sampling memory/time budget exceeded')
                del out, batch, selected_logps, prefix, attached
    finally:
        handle.remove()
    if lora_hash(model) != plan['lora_sha256']:
        raise ValueError('Read-only sampling changed adapter')
    clear = sum(g['signal']['clear_reward_pair'] for g in groups)
    result = {'status': 'completed', 'training_updates': 0, 'questions': len(groups), 'sample_count': 8 * len(groups),
        'clear_reward_pair_groups': clear, 'signal_gate_passed': clear >= plan['minimum_clear_groups'],
        'minimum_clear_groups': plan['minimum_clear_groups'],
        'nonzero_variance_groups': sum(g['signal']['nonzero_reward_variance'] for g in groups),
        'all_zero_groups': sum(all(s['reward'] == 0 for s in g['samples']) for g in groups),
        'all_full_groups': sum(all(s['reward'] >= .999999 for s in g['samples']) for g in groups),
        'length_censored_samples': sum(not s['terminated'] for g in groups for s in g['samples']),
        'canonical_path_differences': sum(s['canonical_path_differs'] for g in groups for s in g['samples']),
        'full_score_samples': sum(s['reward'] >= .999999 for g in groups for s in g['samples']),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'active_seconds_this_process': time.monotonic() - started, 'lora_unchanged': True,
        'scope': 'Training-cohort reward-support feasibility; no best-of8 deployment or model capability improvement.',
        'next': 'Review new sampled semantic pairs and preregister matched SFT/online study before any update.' if clear >= 16
                else 'Stop this sampling configuration; do not increase group size/temperature/seed to rescue the gate.'}
    write_json(ROOT / 'COMPLETION.json', result)
    RESULT_DOC.write_text('# 原始策略同题采样：完成（0训练）\n\n'
        f'32题×8次={result["sample_count"]}条；清晰正负组{clear}/32，事前必要条件>=16，通过={result["signal_gate_passed"]}。'
        f'有方差{result["nonzero_variance_groups"]}组，全零{result["all_zero_groups"]}组，全满分{result["all_full_groups"]}组；'
        f'长度删失{result["length_censored_samples"]}条，实际token与文本重编码不同{result["canonical_path_differences"]}条。\n\n'
        f'峰值allocated {result["peak_allocated_gib"]:.3f}GiB；本进程含加载活动{result["active_seconds_this_process"]:.1f}秒，排队时间另计。'
        '0优化器更新、无新adapter，原LoRA哈希保持。上述是固定训练题上的信号，不是独立泛化或新模型提分。\n\n'
        '奖励沿用原十人VQA共识；满分并不保证所有低分替代都语义错误，新正负需逐图核实。组内相同奖励没有区分优势；'
        '差异足够也不保证策略优化比强SFT更好。不能自动训练、续训负Final或改写旧512联合失败。\n\n'
        '[完整采样JSON](../../../outputs/vlm_sampled_reward_signal_v1/GROUPS.json) · '
        '[固定协议](VLM_SAMPLED_REWARD_SIGNAL_V1_PROTOCOL.md) · '
        '[32原图/奖励/优势学习页](learning/同题采样与GRPO信号_真实token轨迹.html)\n')
    render(result)
    print(result, flush=True)


def controller():
    from research_runtime.queue import run_queue
    while True:
        status = read(NUMERIC / 'queue/STATUS.json')
        if status['status'] == 'failed':
            write_json(ROOT / 'WORKFLOW_STATUS.json', {'status': 'predecessor_failed', 'training_updates': 0})
            raise RuntimeError('Numeric isolation failed; inspect before proceeding')
        if status['status'] == 'completed' and (NUMERIC / 'COMPLETION.json').exists():
            break
        write_json(ROOT / 'WORKFLOW_STATUS.json', {'status': 'waiting_numeric_isolation', 'controller_pid': os.getpid(),
            'predecessor_status': status['status'], 'training_updates': 0, 'existing_tasks_untouched': True})
        time.sleep(15)
    write_json(ROOT / 'WORKFLOW_STATUS.json', {'status': 'waiting_shared_gpu_admission', 'controller_pid': os.getpid(), 'training_updates': 0})
    run_queue(ROOT / 'queue', [{'key': 'sample', 'command': [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
        'completion_marker': str(ROOT / 'COMPLETION.json')}], {'plan_sha256': sha_file(ROOT / 'PLAN.json')})
    write_json(ROOT / 'WORKFLOW_STATUS.json', {'status': 'completed', 'controller_pid': os.getpid(), 'training_updates': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker()
    else:
        controller()
