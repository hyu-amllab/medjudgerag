#!/usr/bin/env python3
"""
merge_lora_adapter.py — Merge a LoRA adapter into a base model (bf16).

Produces a standalone bf16 model for fast inference (no PeftModel overhead).

Usage:
    python src/merge_lora_adapter.py --gpu 0 --base_model <base-model> --adapter_path <lora-adapter> --output_dir outputs/merged_model
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model (bf16)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True,
                        help="Path to LoRA adapter")
    parser.add_argument("--output_dir", default="outputs/merged_model")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model} (bf16, GPU {args.gpu})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Verify
    param = next(model.parameters())
    print(f"\nDone!")
    print(f"  dtype: {param.dtype}")
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  output: {output_dir}")


if __name__ == "__main__":
    main()
