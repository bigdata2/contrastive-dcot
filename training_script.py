import argparse
import collections
import glob
import json
import math
import os
import random

import nltk
import torch

# Compatibility shim: bitsandbytes >=0.43 removed MatmulLtState.memory_efficient_backward
# which peft==0.10.0 reads during 8-bit LoRA setup. The flag defaulted to False and was
# never functionally used for standard QLoRA — restoring it unblocks 8-bit on CUDA 12.4.
try:
    from bitsandbytes.functional import MatmulLtState
    if not hasattr(MatmulLtState, "memory_efficient_backward"):
        MatmulLtState.memory_efficient_backward = False
except Exception:
    pass
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm, trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer

from src.data_processors import DataProcessor, DataProcessorMode
from src.contrastive_trainer import ContrastiveCollator, ContrastiveTrainer


def train(train_hf, tokenizer, ARGS):
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    fp16 = False
    bf16 = False
    per_device_train_batch_size = ARGS.training_batch_size
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    max_seq_length = 4096
    packing = False

    # #Load Datasets

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.enable_input_require_grads()  # required for gradient flow through frozen weights + gradient checkpointing

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=ARGS.lora_path,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=int(len(train_hf)/ARGS.training_batch_size / ARGS.epochs),
        logging_strategy="steps",
        logging_steps=10,
        gradient_checkpointing=gradient_checkpointing,
    )
    print("Save every ", int(len(train_hf)/ARGS.training_batch_size / ARGS.epochs), " steps")
    if ARGS.contrastive:
        # Custom path: plain Trainer + masked NLL + unlikelihood loss.
        # SFTTrainer can't be used here because its forward pass computes a
        # uniform NLL on the whole sequence, which is incompatible with our
        # per-token masking (positive vs. unlikelihood positions).
        from peft import get_peft_model

        # Apply LoRA explicitly (SFTTrainer normally does this for us).
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        collator = ContrastiveCollator(
            tokenizer=tokenizer, max_length=max_seq_length
        )
        # group_by_length needs a `length` column; disable for the contrastive
        # path to avoid having to compute it. The slowdown is minor.
        training_arguments.group_by_length = False
        # remove_unused_columns drops `prompt`/`response`/`neg_span` before the
        # collator sees them; turn it off.
        training_arguments.remove_unused_columns = False

        trainer = ContrastiveTrainer(
            model=model,
            train_dataset=train_hf,
            tokenizer=tokenizer,
            args=training_arguments,
            data_collator=collator,
            alpha=ARGS.alpha,
        )
    else:
        # Original SFTTrainer path (unchanged).
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_hf,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing,
        )
    print("Training started...")
    trainer.train()
    trainer.model.save_pretrained(ARGS.lora_path)
    if ARGS.merge_weights:
        model = trainer.model.merge_and_unload()
        model.save_pretrained(os.path.join(ARGS.lora_path, "merged_model"))
        tokenizer.save_pretrained(os.path.join(ARGS.lora_path, "merged_model"))
    return trainer.model


def get_training_set(ARGS, eos_token):
    mode = None
    if ARGS.contrastive:
        mode = DataProcessorMode.CONTRASTIVE
    elif ARGS.dcot:
        mode = DataProcessorMode.DCOT
    elif ARGS.cot:
        mode = DataProcessorMode.COT
    else:
        raise Exception("Need to set one of these modes: DCoT, CoT, contrastive")
    
    dataset_processor = DataProcessor(
        ARGS.train_path,
        mode=mode,
        eos=eos_token,
        epochs=ARGS.epochs,
        seed=ARGS.seed,
        chat_format=ARGS.chat_format,
        neg_k=ARGS.neg_k,
    )
    train_hf = dataset_processor.get_hf_dataset()
    return train_hf

def parse_args():
    """
    Function to parse arguments
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    # model
    parser.add_argument(
        "--base_model_path",
        type=str,
    )
    parser.add_argument(
        "--lora_path",
        type=str,
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--training_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--chat_format", type=str, help="Options: llama_chat_simple, llama_chat_v2, llama_cot_chat, None")
    parser.add_argument("--merge_weights", action="store_true")
    parser.add_argument("--k", type=int, help="Number of chains to generate for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--dcot", action="store_true", help="Divergent CoT")
    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Contrastive within-inference refinement (NLL on correct + "
             "unlikelihood on the wrong-CoT span).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight on the unlikelihood term (0 disables it; ablate over "
             "{0.25, 0.5, 1.0, 2.0}).",
    )
    parser.add_argument(
        "--neg_k",
        type=int,
        default=-1,
        help="Number of incorrect CoTs to sample per question (-1 = all available).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Starting")
    ARGS = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        ARGS.base_model_path, trust_remote_code=True, use_fast=True
    )
    # llama and phi-2 do not include a pad token by default. The contrastive
    # collator needs a real pad_token_id; the original SFT path is fine with
    # eos as pad too.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = None
    if ARGS.train:
        train_hf = get_training_set(ARGS, tokenizer.eos_token)
        model = train(train_hf, tokenizer, ARGS)

