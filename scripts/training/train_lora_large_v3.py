import os, re, sys, math, time, argparse, random
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, jiwer
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="lv3_lora")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--p_normal", type=float, default=0.25)
ap.add_argument("--no_proj", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--eval_n", type=int, default=500)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--skip_eval0", action="store_true")
ap.add_argument("--save_steps", type=int, default=200)
args = ap.parse_args()

MODEL = "openai/whisper-large-v3"
MANIFEST = "outputs/manifest_all_16k.csv"
CK = os.path.join("checkpoints", args.name)
os.makedirs(CK, exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
random.seed(0); np.random.seed(0); torch.manual_seed(0)

try:
    from opencc import OpenCC
    t2s = OpenCC("t2s").convert
except Exception:
    print("WARNING: no opencc"); t2s = lambda s: s
def norm(s): return re.sub(r"[\W_]", "", t2s(str(s)))
def clean(s): return re.sub(r"\s+", "", str(s))

proc = WhisperProcessor.from_pretrained(MODEL)
fe, tok = proc.feature_extractor, proc.tokenizer
tok.set_prefix_tokens(language="chinese", task="transcribe")

df = pd.read_csv(MANIFEST)
tr = df[(df.split == "train") & df.usable_audio].reset_index(drop=True)
va = df[(df.split == "valid") & df.usable_audio].reset_index(drop=True)
if args.limit: tr = tr.sample(args.limit, random_state=0).reset_index(drop=True)
va = va.iloc[np.random.RandomState(0).choice(len(va), min(args.eval_n, len(va)), replace=False)].reset_index(drop=True)
print("train clips:", len(tr), "| eval subset:", len(va), flush=True)

TR = list(zip(tr.whisper_16k, tr.normal_16k, tr.transcript))

class DS(Dataset):
    def __len__(self): return len(TR)
    def __getitem__(self, i):
        w, n, t = TR[i]
        path = n if random.random() < args.p_normal else w
        x, _ = sf.read(path, dtype="float32")
        f = fe(x, sampling_rate=16000, return_tensors="np").input_features[0]
        return f, tok(clean(t)).input_ids[:448]

hf = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
START = hf.config.decoder_start_token_id
cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
pm = get_peft_model(hf, cfg)          # LoRA layers are injected into hf in place
for p in pm.parameters():
    if p.requires_grad: p.data = p.data.float()
hf.config.use_cache = False
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

proj = None
if not args.no_proj:
    proj = nn.Conv1d(128, 128, 1).to(dev)
    nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)

def adapt(f): return f if proj is None else f + proj(f)

lora_params = [p for p in pm.parameters() if p.requires_grad]
groups = [{"params": lora_params, "lr": args.lr, "weight_decay": 0.01}]
if proj is not None: groups.append({"params": list(proj.parameters()), "lr": args.lr * 3, "weight_decay": 0.0})
opt = torch.optim.AdamW(groups)
print("trainable params: %.1fM" % (sum(p.numel() for g in groups for p in g["params"]) / 1e6), flush=True)

B, A = args.batch, args.accum
spe = len(TR) // (B * A)              # optimizer steps per epoch
total = args.epochs * spe
warm = min(100, max(1, total // 10))
def lr_lambda(s):
    if s < warm: return (s + 1) / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def trainable_state():
    d = {"lora": {n: p.detach().cpu() for n, p in pm.named_parameters() if p.requires_grad}}
    if proj is not None: d["proj"] = proj.state_dict()
    return d

def save_last(step, best):
    torch.save({"state": trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "best": best}, os.path.join(CK, "last.pt.tmp"))
    os.replace(os.path.join(CK, "last.pt.tmp"), os.path.join(CK, "last.pt"))

def save_best():
    d = os.path.join(CK, "best"); os.makedirs(d, exist_ok=True)
    pm.save_pretrained(d)
    if proj is not None: torch.save(proj.state_dict(), os.path.join(d, "proj.pt"))

step, best = 0, 1e9
last = os.path.join(CK, "last.pt")
if os.path.isfile(last):
    ck = torch.load(last, map_location="cpu")
    named = dict(pm.named_parameters())
    for n, v in ck["state"]["lora"].items(): named[n].data.copy_(v.to(dev))
    if proj is not None: proj.load_state_dict(ck["state"]["proj"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    step, best = ck["step"], ck["best"]
    print("RESUMED at step", step, "best", best, flush=True)

@torch.no_grad()
def evaluate():
    hf.eval(); hyps = []
    for i in range(0, len(va), 16):
        xs = [sf.read(p, dtype="float32")[0] for p in va.whisper_16k[i:i + 16]]
        f = torch.from_numpy(fe(xs, sampling_rate=16000, return_tensors="np").input_features).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = hf.generate(input_features=adapt(f), language="chinese", task="transcribe",
                              max_new_tokens=128, num_beams=1, use_cache=True)
        hyps += tok.batch_decode(ids, skip_special_tokens=True)
    refs = [norm(t) for t in va.transcript]; hyps = [norm(h) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r]
    cer = 100 * jiwer.cer([refs[i] for i in keep], [hyps[i] for i in keep])
    hf.train(); return cer

def collate(b):
    f = torch.from_numpy(np.stack([x[0] for x in b]))
    L = max(len(x[1]) for x in b)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    for i, x in enumerate(b): lab[i, :len(x[1])] = torch.tensor(x[1])
    if (lab[:, 0] == START).all(): lab = lab[:, 1:]
    return f, lab

logf = open(f"logs/{args.name}_steps.csv", "a"); 
if step == 0: logf.write("step,loss,lr,samples_per_s\n")
hf.train()
if step == 0 and not args.skip_eval0:
    c0 = evaluate(); print("STEP 0 eval CER on subset: %.2f%% (should be near the baseline)" % c0, flush=True)
    best = c0

ds = DS(); done = False
t0 = time.time(); run = 0.0; seen = 0
for epoch in range(step // spe, args.epochs):
    perm = np.random.RandomState(1234 + epoch).permutation(len(TR))
    batches = [perm[i:i + B].tolist() for i in range(0, spe * A * B, B)]
    skip = (step % spe) * A
    loader = DataLoader(ds, batch_sampler=batches[skip:], num_workers=args.workers,
                        collate_fn=collate, prefetch_factor=4)
    for k, (f, lab) in enumerate(loader):
        f, lab = f.to(dev), lab.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = hf(input_features=adapt(f), labels=lab).loss / A
        loss.backward()
        run += loss.item(); seen += f.shape[0]
        if (k + 1) % A == 0:
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0 or step == 1:
                sps = seen / (time.time() - t0)
                print("ep %d step %d/%d loss %.4f lr %.2e %.1f samp/s  peakmem %.1fGB" % (
                    epoch, step, total, run / (1 if step == 1 else 10), sched.get_last_lr()[0], sps,
                    torch.cuda.max_memory_allocated() / 1e9), flush=True)
                logf.write("%d,%.5f,%.3e,%.2f\n" % (step, run / (1 if step == 1 else 10), sched.get_last_lr()[0], sps)); logf.flush()
                run = 0.0; seen = 0; t0 = time.time()
            if step % args.save_steps == 0: save_last(step, best)
            if args.max_steps and step >= args.max_steps: done = True; break
    c = evaluate()
    print("=== epoch %d done | step %d | valid-subset CER %.2f%% (best %.2f%%)" % (epoch, step, c, best), flush=True)
    if c < best: best = c; save_best(); print("saved best", flush=True)
    save_last(step, best)
    if done: break
print("FINISHED. best valid-subset CER: %.2f%%" % best)
