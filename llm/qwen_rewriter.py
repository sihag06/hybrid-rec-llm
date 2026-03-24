from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class QwenRewriter:
    """
    Lightweight wrapper around Qwen chat models for structured rewrite.
    Produces a JSON snippet per the provided schema and few-shot examples.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", default_examples: Optional[List[str]] = None) -> None:
        self.model_name = model_name
        self.model, self.tokenizer = _load_model(model_name)
        self.device = self.model.device
        self.default_examples = default_examples or []
        self.system_prompt = (
            "Rewrite the job description into a concise retrieval query and a full rerank query. "
            "Follow the JSON template exactly. If a field is unknown, set it to null or an empty list."
            "You are an expert at classifying assessment intent. "
            "For 'intent' field, output EX  ACTLY one of: TECH, BEHAVIORAL, MIXED, UNKNOWN. "
            "TECH = technical skills, BEHAVIORAL = soft skills/leadership/sales/personality, "
            "MIXED = both, UNKNOWN = unclear. Never output the list itself."
        )

    def predict(
        self,
        text: str,
        schema: str,
        examples: Optional[List[str]] = None,
        max_length: int = 4000,
        return_full: bool = False,
    ):
        examples = examples or self.default_examples
        schema_fmt = json.dumps(json.loads(schema), indent=4)
        prompt = "### Template:\n" + schema_fmt + "\n"
        for ex in examples:
            if ex:
                prompt += "### Example:\n" + json.dumps(json.loads(ex), indent=4) + "\n"
        prompt += "### Text:\n" + text + "\n### Output JSON:\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([chat], return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        # remove the prompt tokens
        gen_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
        output = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        cleaned = output.strip()
        cleaned = cleaned.replace("MAX|TARGET|null", "null").replace("int|null", "null")
        cleaned = cleaned.replace("string", "\"\"")
        print (cleaned) 
        if return_full:
            return {"prompt": prompt, "raw_output": output, "clean_output": cleaned}
        return cleaned
