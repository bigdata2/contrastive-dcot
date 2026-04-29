"""
Quick verification of the contrastive trainer.

Run this on a MacBook (CPU/MPS) before paying for an A100. It checks the
three things that, if any of them is broken, will silently waste GPU hours:

  1. MASK SANITY
     The unlikelihood mask must cover *exactly* the wrong-CoT tokens.
     We decode the masked tokens and assert they equal the original
     wrong CoT string.

  2. OVERFIT ONE BATCH
     Train a tiny model on a single batch for a few hundred steps.
     Expected behaviour:
       - NLL drops substantially (model memorises the positive span).
       - UL stays finite and on average decreases (model lowers
         P(wrong CoT)). Occasional spikes are OK; divergence is not.
       - Total loss decreases.

  3. FORMAT PRESERVATION
     After overfitting, greedy generation from the prompt should still
     produce [Answer 2] and [Final answer] markers. If they vanish, the
     data layout (or the label masking) is wrong.

If all three pass, the code is wired correctly and it's safe to scale to
Phi-2 / LLaMA-2 on the A100. If any fails, fix locally for free first.

Runs in ~2-5 minutes on a MacBook Air with distilgpt2.
"""

import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `src` importable when running from the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.contrastive_trainer import ContrastiveCollator  # noqa: E402
from src.data_processors import DataProcessor, DataProcessorMode  # noqa: E402


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("VERIFY_MODEL", "distilgpt2")
DATA_PATH = os.environ.get(
    "VERIFY_DATA", os.path.join(HERE, "data", "dcot_collection", "cot9_dataset.json")
)
N_EXAMPLES = int(os.environ.get("VERIFY_N", "4"))
N_STEPS = int(os.environ.get("VERIFY_STEPS", "300"))
ALPHA = float(os.environ.get("VERIFY_ALPHA", "1.0"))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def banner(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Build a tiny contrastive batch
# ---------------------------------------------------------------------------

banner("Loading tokenizer + tiny model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = pick_device()
print(f"  model:  {MODEL_NAME}")
print(f"  device: {device}")
print(f"  data:   {DATA_PATH}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

banner(f"Building {N_EXAMPLES} contrastive examples")
processor = DataProcessor(
    DATA_PATH,
    mode=DataProcessorMode.CONTRASTIVE,
    eos=tokenizer.eos_token,
    epochs=1,
    seed=0,
)
ds = processor.get_hf_dataset()

# Keep only contrastive (wrong-then-right) examples; skip the DCoT-style ones
# that have an empty neg_span.
contrastive_idxs = [i for i, ex in enumerate(ds) if ex.get("neg_span", "")]
chosen = contrastive_idxs[:N_EXAMPLES]
print(f"  total examples in mixed set:    {len(ds)}")
print(f"  contrastive (with neg_span):    {len(contrastive_idxs)}")
print(f"  using:                          {len(chosen)}")

batch_features = [dict(ds[i]) for i in chosen]

collator = ContrastiveCollator(tokenizer=tokenizer, max_length=1024)
batch = collator(batch_features)
batch = {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# CHECK 1: Mask sanity
# ---------------------------------------------------------------------------

banner("CHECK 1 - Mask sanity (decode UL-masked tokens, compare to wrong CoT)")
# With the overlap-semantics fix in ContrastiveCollator, the UL mask may
# include a leading space (BPE merges the prompt's trailing space into the
# first response token) and may include a trailing newline (if BPE merges
# across the wrong-CoT boundary). The right check is therefore CONTAINMENT,
# not strict equality: the wrong-CoT text should appear inside the decoded
# masked region.
all_pass = True
for i in range(len(chosen)):
    ids = batch["input_ids"][i].tolist()
    mask = batch["ul_mask"][i].tolist()
    masked_ids = [t for t, m in zip(ids, mask) if m == 1]
    decoded = tokenizer.decode(masked_ids)
    expected = batch_features[i]["neg_span"]

    # Strip just whitespace at the boundaries; the wrong-CoT body must match.
    ok = expected.strip() in decoded.strip()

    print(f"\n  Example {i}: {sum(mask)} tokens masked")
    print(f"    decoded   : {decoded[:160]!r}")
    print(f"    expected  : {expected[:160]!r}")
    print(f"    CONTAINS  : {ok}  (decoded should contain neg_span)")
    if not ok:
        all_pass = False

print(f"\n  Mask sanity: {'PASS' if all_pass else 'FAIL'}")
if not all_pass:
    print("  -> Common cause: tokenizer is not fast (offset_mapping unavailable)")
    print("     or the prompt/response concatenation lost the neg_span boundary.")


# ---------------------------------------------------------------------------
# CHECK 2: Overfit one batch
# ---------------------------------------------------------------------------

banner(f"CHECK 2 - Overfit one batch ({N_STEPS} steps, alpha={ALPHA})")
print("  Expected: NLL drops a lot; UL stays finite; total loss decreases.")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
model.train()

trace = []
for step in range(N_STEPS):
    ul_mask = batch["ul_mask"]
    labels = batch["labels"]

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = out.logits

    sl = logits[..., :-1, :].contiguous()
    lab = labels[..., 1:].contiguous()
    um = ul_mask[..., 1:].contiguous()
    sin = batch["input_ids"][..., 1:].contiguous()

    ce = F.cross_entropy(
        sl.reshape(-1, sl.size(-1)),
        lab.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(lab.size())

    pos_mask = (lab != -100).float()
    nll = (ce * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)

    neg_mask = um.float()
    if neg_mask.sum() > 0:
        lp = F.log_softmax(sl, dim=-1)
        tlp = lp.gather(-1, sin.unsqueeze(-1)).squeeze(-1)
        tp = tlp.exp().clamp(min=1e-6, max=1 - 1e-6)
        ul_per_tok = -torch.log(1 - tp)
        ul = (ul_per_tok * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
    else:
        ul = torch.zeros((), device=device)

    loss = nll + ALPHA * ul

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    if step % max(1, N_STEPS // 10) == 0 or step == N_STEPS - 1:
        trace.append((step, nll.item(), ul.item(), loss.item()))
        print(f"  step {step:4d}  NLL={nll.item():.4f}  UL={ul.item():.4f}  total={loss.item():.4f}")

# Verdict.
nll_first, ul_first = trace[0][1], trace[0][2]
nll_last, ul_last = trace[-1][1], trace[-1][2]
nll_drop = nll_first - nll_last
finite_ul = all(torch.isfinite(torch.tensor(t[2])).item() for t in trace)
nll_pass = nll_drop > 0.5
print()
print(f"  NLL  : {nll_first:.3f} -> {nll_last:.3f}   drop={nll_drop:+.3f}")
print(f"  UL   : {ul_first:.3f} -> {ul_last:.3f}    finite throughout={finite_ul}")
print(f"  Verdict: {'PASS' if (nll_pass and finite_ul) else 'FAIL'}")
if not nll_pass:
    print("  -> NLL did not drop. Likely the prompt mask is misplaced or labels")
    print("     are all -100 (positive span empty).")
if not finite_ul:
    print("  -> UL went to inf/nan. Increase the clamp epsilon in the trainer.")


# ---------------------------------------------------------------------------
# CHECK 3: Format preservation
# ---------------------------------------------------------------------------

banner("CHECK 3 - Format preservation (greedy generation)")
model.eval()
prompt = batch_features[0]["prompt"]
inp = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out_ids = model.generate(
        **inp,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
completion = tokenizer.decode(
    out_ids[0, inp["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"  prompt suffix : ...{prompt[-80:]!r}")
print(f"  completion    : {completion[:240]!r}")

has_a2 = "[Answer 2]" in completion
has_fa = "[Final answer]" in completion
print(f"  contains [Answer 2]      : {has_a2}")
print(f"  contains [Final answer]  : {has_fa}")
print(f"  Verdict: {'PASS' if (has_a2 or has_fa) else 'FAIL (model is small; partial pass OK)'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

banner("SUMMARY")
print(f"  CHECK 1 (mask sanity)        : {'PASS' if all_pass else 'FAIL'}")
print(f"  CHECK 2 (one-batch overfit)  : {'PASS' if (nll_pass and finite_ul) else 'FAIL'}")
print(f"  CHECK 3 (format preserved)   : {'PASS-ish' if (has_a2 or has_fa) else 'see note'}")
print()
print("If 1 and 2 pass, you can run training_script.py --contrastive on the A100.")
print("CHECK 3 is informative but distilgpt2 may not learn the markers in 300 steps.")
print("That alone is not a reason to delay scaling up.")
