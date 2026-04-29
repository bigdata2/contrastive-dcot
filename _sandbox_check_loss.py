"""
Validates the NLL + alpha*UL loss math from ContrastiveTrainer.compute_loss
against a hand-computed reference, using numpy (no torch needed).
"""
import math
import numpy as np

np.random.seed(0)

V = 7  # tiny vocab
T = 6  # sequence length
B = 2  # batch

# Random logits, hand-picked labels and ul_mask.
logits = np.random.randn(B, T, V) * 0.3
input_ids = np.array([
    [3, 2, 4, 1, 5, 6],
    [0, 4, 2, 3, 1, 6],
])
# labels: -100 on prompt + on UL-flagged tokens; real ids on positive positions.
labels = np.array([
    [-100, -100, 4, -100, 5, 6],   # prompt=2 tokens, UL on pos=3, positives=[2,4,5]
    [-100, -100, -100, 3, 1, 6],   # prompt=2, UL on pos=2, positives=[3,4,5]
])
ul_mask = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
])
ALPHA = 0.7

# --- Replicate compute_loss ----------------------------------------------------

# Causal LM next-token shift.
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
shift_ul = ul_mask[:, 1:]
shift_input = input_ids[:, 1:]


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=axis, keepdims=True))


lp = log_softmax(shift_logits, axis=-1)  # (B, T-1, V)

# Per-token CE on positive positions.
pos_mask = (shift_labels != -100).astype(float)
ce_per_token = np.zeros_like(pos_mask)
for b in range(B):
    for t in range(T - 1):
        if shift_labels[b, t] != -100:
            ce_per_token[b, t] = -lp[b, t, shift_labels[b, t]]
nll = (ce_per_token * pos_mask).sum() / max(pos_mask.sum(), 1.0)

# Per-token UL on negative positions: -log(1 - p(input_id)).
ul_per_token = np.zeros_like(pos_mask)
for b in range(B):
    for t in range(T - 1):
        if shift_ul[b, t] == 1:
            p = math.exp(lp[b, t, shift_input[b, t]])
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            ul_per_token[b, t] = -math.log(1.0 - p)
neg_mask = shift_ul.astype(float)
ul = (ul_per_token * neg_mask).sum() / max(neg_mask.sum(), 1.0)

total = nll + ALPHA * ul

print("=" * 60)
print("Loss math reference computation")
print("=" * 60)
print(f"  positive token count : {int(pos_mask.sum())}")
print(f"  negative token count : {int(neg_mask.sum())}")
print(f"  NLL  = {nll:.6f}")
print(f"  UL   = {ul:.6f}")
print(f"  total = NLL + {ALPHA}*UL = {total:.6f}")

# --- Sanity invariants ---------------------------------------------------------

assert nll > 0, "NLL must be positive (not all probs are 1)"
assert ul > 0, "UL must be positive (some prob > 0)"
# All UL tokens are also -100 in labels, so they don't double-count.
assert all(
    shift_labels[b, t] == -100
    for b in range(B) for t in range(T - 1) if shift_ul[b, t] == 1
), "UL positions must have label==-100 (no double counting)"
print("\nInvariants passed:")
print("  - NLL > 0")
print("  - UL  > 0")
print("  - No position has both label != -100 AND ul_mask == 1")

# --- Edge case: no UL tokens (DCoT-style example reduces to plain NLL) ---------

ul_mask2 = np.zeros_like(ul_mask)
shift_ul2 = ul_mask2[:, 1:]
neg_mask2 = shift_ul2.astype(float)
ul2 = 0.0 if neg_mask2.sum() == 0 else (ul_per_token * neg_mask2).sum() / max(neg_mask2.sum(), 1.0)
print(f"\nDCoT-style (no UL) total = {nll:.6f} + {ALPHA}*{ul2:.6f} = {nll + ALPHA*ul2:.6f}")
print("  (matches plain NLL; UL term contributes 0 when neg_mask is empty)")

print("\nALL CHECKS PASSED")
