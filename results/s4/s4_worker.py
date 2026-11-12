
import sys, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
shard, cache, model_dir, log_path, max_new, max_bs, budget = sys.argv[1:8]
max_new, max_bs, budget = int(max_new), int(max_bs), int(budget)
log = open(log_path, "a", buffering=1)
def say(*a): print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)
done = set()
try:
    for l in open(cache):
        done.add(json.loads(l)["h"])
except FileNotFoundError:
    pass
items = [json.loads(l) for l in open(shard)]
items = [it for it in items if it["h"] not in done]
say(f"{len(items)} prompts to generate ({len(done)} cached)")
if not items:
    sys.exit(0)
tok = AutoTokenizer.from_pretrained(model_dir)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
import transformers
kw = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch.float16}
model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=q, device_map={"": 0},
                                             attn_implementation="sdpa", **kw).eval()
say("model loaded:", torch.cuda.get_device_name(0), f"{torch.cuda.memory_allocated()/2**30:.1f} GiB")
texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in items]
lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
order = sorted(range(len(items)), key=lambda i: (-lens[i], items[i]["h"]))   # longest first: OOM shows up early
batches, cur = [], []
for i in order:
    if cur and (len(cur) + 1 > max_bs or (len(cur) + 1) * lens[cur[0]] > budget):
        batches.append(cur); cur = []
    cur.append(i)
if cur:
    batches.append(cur)
out = open(cache, "a", buffering=1)

@torch.inference_mode()
def run(idx):
    enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
    try:
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
    except torch.cuda.OutOfMemoryError:
        del enc
        torch.cuda.empty_cache()
        if len(idx) == 1:
            raise
        h = len(idx) // 2
        run(idx[:h]); run(idx[h:])
        return
    dec = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for i, d in zip(idx, dec):
        out.write(json.dumps({"h": items[i]["h"], "out": d, "in_tok": lens[i]}, ensure_ascii=False) + "\n")

t0, n = time.time(), 0
for b, idx in enumerate(batches):
    run(idx)
    n += len(idx)
    if b % 10 == 0 or b == len(batches) - 1:
        say(f"{n}/{len(items)} ({n/(time.time()-t0):.2f}/s, batch {len(idx)} x {lens[idx[0]]} tok)")
say("done")
