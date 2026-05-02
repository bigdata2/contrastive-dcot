"""
Contrastive trainer for DCoT.

Implements Option 2 of the contrastive within-inference refinement recipe:
  - Standard NLL on every response token EXCEPT the wrong-CoT span.
  - Unlikelihood loss on the wrong-CoT span (Welleck et al., 2020):
        L_UL = - log(1 - p_theta(y_t | y_<t))
  - Total loss: L = L_NLL + alpha * L_UL.

Examples without a `neg_span` (the DCoT-style ones built by
`create_contrastive_dataset`) reduce to ordinary NLL with no unlikelihood
contribution -- this is what preserves DCoT's existing capability.

The collator uses character-level offset_mapping (requires a fast
tokenizer) so the unlikelihood mask is robust to BPE boundaries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, Trainer


@dataclass
class ContrastiveCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    # Right-pad for training (standard for causal LM SFT). Left-padding is only
    # needed at generation time, which is handled by evaluation.py separately.
    padding_side: str = "right"

    def __post_init__(self):
        if not self.tokenizer.is_fast:
            raise ValueError(
                "ContrastiveCollator requires a fast tokenizer "
                "(use_fast=True) so that offset_mapping is available. "
                "Pass AutoTokenizer.from_pretrained(..., use_fast=True)."
            )
        # ensure a pad token id exists
        self._pad_id = self.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode_one(self, prompt: str, response: str, neg_span: str):
        full_text = prompt + response

        # Tokenize the full sequence with character offsets.
        enc = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids: List[int] = enc["input_ids"]
        offsets: List[tuple] = enc["offset_mapping"]
        T = len(input_ids)

        # Locate where the response begins in *characters*.
        prompt_char_len = len(prompt)

        # Mask the prompt out of the loss. We use overlap semantics
        # (`e > prompt_char_len`) rather than strict containment
        # (`s >= prompt_char_len`) because BPE often merges a trailing space
        # in the prompt into the first response token. Such a token's start
        # offset is *before* prompt_char_len but its content is mostly
        # response, so we want it supervised.
        labels = [self.label_pad_token_id] * T
        for i, (s, e) in enumerate(offsets):
            # Special tokens often have (0, 0) offsets. Treat them as prompt.
            if s == 0 and e == 0 and i < T - 1:
                continue
            if e > prompt_char_len:
                labels[i] = input_ids[i]

        # Build the unlikelihood mask: tokens that overlap with the neg_span
        # character range. Same overlap rationale as the prompt boundary --
        # the first wrong-CoT token often straddles the prompt boundary
        # because of the leading-space BPE merge, and the last wrong-CoT
        # token may straddle the response boundary if BPE merges across the
        # trailing newline. Including both keeps the unlikelihood pressure
        # complete.
        ul_mask = [0] * T
        if neg_span:
            neg_start = full_text.find(neg_span)
            if neg_start != -1:
                neg_end = neg_start + len(neg_span)
                for i, (s, e) in enumerate(offsets):
                    if s == 0 and e == 0:
                        continue
                    # Any overlap with [neg_start, neg_end).
                    if e > neg_start and s < neg_end:
                        ul_mask[i] = 1
                # NLL and UL co-exist at neg_span positions per Welleck et al. (2019)
                # Eq. (4): L = -log p(x*) [NLL on correct token] + UL on negative candidates.
                # Zeroing labels here creates a void with no positive gradient, which causes
                # the model to fall into degenerate attractors ([[[[) because UL without NLL
                # has no target to anchor generation. Keeping labels intact means the wrong-CoT
                # tokens get NLL (model learns to generate fluent Answer-1 text) AND UL
                # (those same tokens are gently suppressed). The two gradients partially cancel,
                # resulting in reduced-but-nonzero probability for wrong-CoT patterns.
                # This is the paper's intent: maintain fluency while discouraging known-wrong tokens.

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * T,
            "labels": labels,
            "ul_mask": ul_mask,
        }

    def _pad(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        padded = []
        for s in sequences:
            pad_n = max_len - len(s)
            if self.padding_side == "right":
                padded.append(list(s) + [pad_value] * pad_n)
            else:
                padded.append([pad_value] * pad_n + list(s))
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = [
            self._encode_one(
                f["prompt"], f["response"], f.get("neg_span", "") or ""
            )
            for f in features
        ]

        return {
            "input_ids": self._pad([e["input_ids"] for e in encoded], self._pad_id),
            "attention_mask": self._pad(
                [e["attention_mask"] for e in encoded], 0
            ),
            "labels": self._pad(
                [e["labels"] for e in encoded], self.label_pad_token_id
            ),
            "ul_mask": self._pad([e["ul_mask"] for e in encoded], 0),
        }


class ContrastiveTrainer(Trainer):
    """
    Trainer that adds an unlikelihood term on positions flagged by `ul_mask`.

      L = mean_{t in P_pos} -log p(y_t | y_<t)
        + alpha * mean_{t in P_neg} -log(1 - p(y_t | y_<t))

    P_pos is positions with labels != -100 (set by the collator).
    P_neg is positions with ul_mask == 1.
    The collator zeroes the labels on P_neg so they don't contribute to NLL.
    """

    def __init__(self, *args, alpha: float = 1.0, log_components: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.log_components = log_components
        self._last_nll: float = 0.0
        self._last_ul: float = 0.0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        ul_mask = inputs.pop("ul_mask", None)
        labels = inputs["labels"]

        # Forward pass without letting the model compute its own loss.
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, T, V]

        # Causal-LM next-token shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if ul_mask is not None:
            shift_ul = ul_mask[..., 1:].contiguous()
        else:
            shift_ul = torch.zeros_like(shift_labels)

        # Cast to float32 before any loss computation.
        # 8-bit (and 4-bit) quantized models produce logits in float16; with
        # phi-2's 51 200-token vocabulary, softmax over float16 overflows to NaN.
        # Upcasting here fixes the NaN without changing the backward graph because
        # the cast is outside the model's own compute graph.
        shift_logits = shift_logits.float()

        # Per-token NLL (positions where labels != -100).
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())

        pos_mask = (shift_labels != -100).float()
        nll = (ce * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)

        # Unlikelihood on positions flagged by ul_mask. We need the original
        # token id at those positions (the "wrong CoT" token), which we read
        # off the input sequence rather than `labels` (which the collator
        # zeroed out for those positions).
        neg_mask = shift_ul.float()
        if neg_mask.sum() > 0:
            shift_input_ids = inputs["input_ids"][..., 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)  # already float32
            target_lp = log_probs.gather(
                -1, shift_input_ids.unsqueeze(-1)
            ).squeeze(-1)
            target_p = target_lp.exp().clamp(min=1e-6, max=1.0 - 1e-6)
            ul_per_token = -torch.log(1.0 - target_p)
            ul = (ul_per_token * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
        else:
            ul = torch.zeros((), device=nll.device, dtype=nll.dtype)

        loss = nll + self.alpha * ul

        # Stash for optional logging.
        self._last_nll = float(nll.detach().item())
        self._last_ul = float(ul.detach().item()) if isinstance(ul, torch.Tensor) else 0.0

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):  # type: ignore[override]
        if self.log_components:
            logs = dict(logs)
            logs.setdefault("nll", self._last_nll)
            logs.setdefault("ul", self._last_ul)
        return super().log(logs, *args, **kwargs)
