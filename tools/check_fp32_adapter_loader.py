"""One real-model regression for the new checkpoint-preserving loader."""
import json
from pathlib import Path
import sys
from diagnose_sft_failure import OUT,read,setup


def worker():
    torch,c,m=setup()
    from safetensors.torch import load_file
    from vlm_fp32_adapter.model import load
    from vlm_probe.evaluate import evaluate_batch,processor_for
    from vlm_deep.common import write_json
    adapter=Path(c['output_root'])/'high1024/adapter'
    model,binding=load(c['model_root'],adapter,torch.device('cuda:0'))
    state=load_file(str(adapter/'adapter_model.safetensors'))
    for name,p in model.named_parameters():
        if 'lora_' in name:
            assert p.dtype==torch.float32 and torch.equal(p.detach().cpu(),state[name.replace('.default.','.')])
        elif p.is_floating_point():assert p.dtype==torch.bfloat16
        assert not p.requires_grad
    dev=m['splits']['dev'];index=next(i for i,r in enumerate(dev) if str(r['question_id'])=='10282');begin=index//4*4
    chunk=dev[begin:begin+4]
    processor=processor_for(Path(c['model_root']),read(c['reference_config']),1024)
    with torch.inference_mode():
        answers,_=evaluate_batch(model,processor,chunk,m['images'],Path(c['manifest']).parent,torch.device('cuda:0'),1024,
            processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),32)
    expected={r['question_id']:r['prediction'] for r in read(OUT/'PRECISION_master_fp32.json')['records']}
    actual={r['question_id']:a['prediction'] for r,a in zip(chunk,answers)}
    assert actual=={r['question_id']:expected[r['question_id']] for r in chunk}
    result={'status':'passed','binding':binding,'actual_predictions':actual,'all144_factors_equal_real_FP32_checkpoint':True,
        'expected_batch_predictions_reproduced':True,'training_mode_backward_tested':False,'optimizer_updates':0,
        'scope':'new loader actual GPU eval integration; not a training or performance-gain claim'}
    write_json(OUT/'FP32_LOADER_GPU_CHECK.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:
        from research_runtime.queue import run_queue
        from vlm_deep.common import sha_file
        run_queue(OUT/'loader_check_queue',[{'key':'loader_check','command':[sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
            'completion_marker':str(OUT/'FP32_LOADER_GPU_CHECK.json')}],{'source_sha256':sha_file(Path(__file__)),'zero_optimizer_updates':True})
