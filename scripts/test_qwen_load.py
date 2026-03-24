from __future__ import annotations

"""
Simple one-off loader to verify Qwen model availability and a basic generation.
Run:
  /usr/bin/python3 scripts/test_qwen_load.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Test prompt"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Test loading Qwen and run a short generation.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="Give me a short introduction to large language models.")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            top_p=0.9,
        )
    # strip input tokens
    gen_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
