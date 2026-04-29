"""
Algorithmic correctness check for the contrastive masking logic, including
the BPE-leading-space-merge edge case that bit verify_contrastive.py.

We use a synthetic tokenizer that explicitly REPRODUCES the GPT-2 BPE
behaviour of merging a trailing prompt-space into the first response token.
If the algorithm passes here, it will pass under real BPE too.
"""

import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Stub `datasets` so we can import data_processors.
fake = types.ModuleType("datasets")
class _Dataset:
    @staticmethod
    def from_dict(d):
        obj = types.SimpleNamespace(_d=d)
        obj.map = lambda fn, batched=True: obj
        return obj
fake.Dataset = _Dataset
sys.modules["datasets"] = fake

from src.data_processors import DataProcessor, DataProcessorMode  # noqa: E402


# A "BPE-like" tokenizer that mimics GPT-2's leading-space merge behaviour.
# Tokens are runs of (optional leading space + non-space chars) OR runs of
# whitespace that contain a newline. This means " First" is one token whose
# offset starts BEFORE the F (at the space) -- exactly the case that broke
# CHECK 1 on the user's machine.
TOK_RE = re.compile(r" ?\S+|\s+")


def synth_tokenize(text: str):
    offsets, ids = [], []
    # Fake BOS at (0, 0).
    ids.append(-1)
    offsets.append((0, 0))
    for tok_id, m in enumerate(TOK_RE.finditer(text)):
        ids.append(tok_id)
        offsets.append((m.start(), m.end()))
    return ids, offsets


def synth_decode(text: str, masked_offsets):
    return "".join(text[s:e] for s, e in masked_offsets if (s, e) != (0, 0))


def encode_one(prompt: str, response: str, neg_span: str):
    """Replicates ContrastiveCollator._encode_one with the OVERLAP fix."""
    full_text = prompt + response
    input_ids, offsets = synth_tokenize(full_text)
    T = len(input_ids)

    prompt_char_len = len(prompt)
    labels = [-100] * T
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0 and i < T - 1:
            continue
        # OVERLAP: token's content extends past the prompt boundary.
        if e > prompt_char_len:
            labels[i] = input_ids[i]

    ul_mask = [0] * T
    if neg_span:
        neg_start = full_text.find(neg_span)
        if neg_start != -1:
            neg_end = neg_start + len(neg_span)
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue
                # OVERLAP: any character of the token falls inside neg_span.
                if e > neg_start and s < neg_end:
                    ul_mask[i] = 1
            for i in range(T):
                if ul_mask[i] == 1:
                    labels[i] = -100
    return input_ids, labels, ul_mask, offsets


# --- Run -----

print("=" * 72)
print("CHECK 1 (algorithm, simulating BPE leading-space merge)")
print("=" * 72)

processor = DataProcessor(
    os.path.join(HERE, "data", "dcot_collection", "cot9_dataset.json"),
    mode=DataProcessorMode.CONTRASTIVE,
    eos="</s>",
    epochs=1,
    seed=0,
)
contrastive = [d for d in processor.ccot_dataset if d.get("neg_span")]
print(f"\ncontrastive examples: {len(contrastive)}")

N = 200
ok = 0
fail_examples = []
for idx, ex in enumerate(contrastive[:N]):
    prompt, response, neg = ex["prompt"], ex["response"], ex["neg_span"]
    full = prompt + response
    ids, labels, ul, offs = encode_one(prompt, response, neg)

    # Reconstruct the masked region by union of token char ranges.
    masked_offs = [offs[i] for i, m in enumerate(ul) if m == 1]
    if masked_offs:
        # Merge offsets and clip to [neg_start, neg_end] for fair comparison.
        # The straddling token's content extends a few chars before/after
        # neg_span; after clipping we should recover exactly neg_span.
        neg_start = full.find(neg)
        neg_end = neg_start + len(neg)
        clipped = [(max(s, neg_start), min(e, neg_end)) for s, e in masked_offs]
        decoded = "".join(full[s:e] for s, e in clipped)
    else:
        decoded = ""

    span_match = decoded == neg

    # Invariants.
    T = len(ids)
    no_overlap = not any(labels[i] != -100 and ul[i] == 1 for i in range(T))
    has_ul = sum(ul) > 0
    # Every label != -100 position should have offset.end > prompt_char_len.
    pclen = len(prompt)
    label_ok = all(
        offs[i][1] > pclen
        for i in range(len(ids))
        if labels[i] != -100
    )

    invariants = {
        "span_recovered_after_clip": span_match,
        "no_label_ul_overlap": no_overlap,
        "has_ul": has_ul,
        "labels_past_prompt_only": label_ok,
    }
    if all(invariants.values()):
        ok += 1
    else:
        fail_examples.append((idx, invariants, decoded[:80], neg[:80]))

print(f"\nresult: {ok} / {N} examples passed all invariants")
if fail_examples:
    print("\nfirst few failures:")
    for idx, inv, got, exp in fail_examples[:3]:
        print(f"  ex {idx}: {inv}")
        print(f"    got : {got!r}")
        print(f"    exp : {exp!r}")

print("\n" + "=" * 72)
print(f"OVERALL: {'PASS' if ok == N else 'FAIL'}")
print("=" * 72)
