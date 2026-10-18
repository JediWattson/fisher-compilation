"""Sequential native/MLX parity probe; run teacher then mlx in separate processes.

Use MLX_ENABLE_TF32=0 for the MLX stage. No weights are downloaded.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=('teacher', 'mlx'))
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--report', type=Path)
args = parser.parse_args()
snapshot, ref = args.snapshot, args.reference
if args.stage == 'teacher':
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,padding_side='left')
    batch=tok(['The capital of France is','A triangle has'],padding=True,return_tensors='pt')
    ids,mask=batch['input_ids'],batch['attention_mask']
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    arrays={'ids':ids.numpy(),'mask':mask.numpy()}
    started=time.perf_counter()
    with torch.inference_mode():
        for step in range(5):
            pos=(mask.cumsum(1)-1).clamp(min=0)
            out=model(input_ids=ids,attention_mask=mask,position_ids=pos,output_hidden_states=True,use_cache=False)
            arrays[f'logits_{step}']=out.logits.numpy()
            if step==0:
                arrays['terminal_normalized']=out.hidden_states[-1].numpy()
                for i in range(model.config.num_hidden_layers): arrays[f'layer_input_{i}']=out.hidden_states[i].numpy()
            nxt=out.logits[:,-1].argmax(-1,keepdim=True)
            arrays[f'token_{step}']=nxt.numpy()
            ids=torch.cat((ids,nxt),1)
            mask=torch.cat((mask,torch.ones_like(nxt)),1)
    np.savez(ref,**arrays)
    print(json.dumps({'teacher_complete':True,'seconds':time.perf_counter()-started,'shape':list(arrays['ids'].shape)}),flush=True)
else:
    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    r=np.load(ref)
    model=MLXQwen3.from_pretrained(snapshot,dtype=mx.float32)
    mx.eval(model.parameters)
    start=time.perf_counter()
    out=model(r['ids'],attention_mask=r['mask'],capture=True)
    mx.eval(out.logits,out.captures)
    prefill_s=time.perf_counter()-start
    mask=r['mask'].astype(bool)
    errors=[float(np.max(np.abs(np.asarray(out.logits)[mask]-r['logits_0'][mask])))]
    hidden=[float(np.max(np.abs(np.asarray(out.captures[f'attention.{i}.input'])[mask]-r[f'layer_input_{i}'][mask]))) for i in range(len(model.layers))]
    terminal=float(np.max(np.abs(np.asarray(out.captures['terminal.normalized'])[mask]-r['terminal_normalized'][mask])))
    argmax=[bool(np.array_equal(np.asarray(out.logits[:,-1]).argmax(-1),r['token_0'].ravel()))]
    times=[]
    for step in range(1,5):
        start=time.perf_counter()
        out=model.decode(mx.array(r[f'token_{step-1}']),out.cache)
        mx.eval(out.logits,out.cache.key_values)
        times.append(time.perf_counter()-start)
        errors.append(float(np.max(np.abs(np.asarray(out.logits[:,-1])-r[f'logits_{step}'][:,-1]))))
        argmax.append(bool(np.array_equal(np.asarray(out.logits[:,-1]).argmax(-1),r[f'token_{step}'].ravel())))
    report={'model':'local dense Qwen3','snapshot':str(snapshot.resolve()),'revision':snapshot.name,'dtype':'float32','batch_shape':list(r['ids'].shape),'teacher':'HF PyTorch eager, sequential process','backend':'MLX Metal','prefill_seconds':prefill_s,'decode_seconds':times,'valid_logit_max_errors':errors,'layer_input_max_error':max(hidden),'terminal_normalized_max_error':terminal,'last_token_argmax_matches':argmax,'scope':'two short prompts and four teacher-forced cached steps; numerical parity smoke, not a quality benchmark'}
    print(json.dumps(report,indent=2),flush=True)
    report['MLX_ENABLE_TF32'] = os.environ.get('MLX_ENABLE_TF32', 'unset')
    report['last_token_decision_count'] = len(argmax) * r['ids'].shape[0]
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    if max(errors) > 0.001 or not all(argmax):
        raise SystemExit('Parity gate failed: max valid logit error > 0.001 or next-token mismatch')
