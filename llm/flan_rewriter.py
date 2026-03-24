from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class FlanRewriter:
    """
    Simple wrapper around FLAN-T5 style models for structured query rewrite.
    Produces JSON per schema and few-shot examples.
    """

    def __init__(self, model_name: str = "google/flan-t5-small", default_examples: Optional[List[str]] = None) -> None:
        self.model_name = model_name
        self.model, self.tokenizer = _load_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.default_examples = default_examples or []
        self.system_prompt = (
            "Rewrite the job description into JSON following the template. "
            "If a field is unknown, use null or empty list. "
            "For 'intent', output exactly one of: TECH, BEHAVIORAL, MIXED, UNKNOWN."
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
        prompt = self.system_prompt + "\n"
        prompt += "### Template:\n" + schema_fmt + "\n"
        for ex in examples:
            if ex:
                prompt += "### Example:\n" + json.dumps(json.loads(ex), indent=4) + "\n"
        prompt += "### Text:\n" + text + "\n### Output JSON:\n"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cleaned = self._extract_json_like(output.strip())
        if not cleaned and output:
            cleaned = output.strip()
        if return_full:
            return {"prompt": prompt, "raw_output": output, "clean_output": cleaned}
        return cleaned

    @staticmethod
    def _extract_json_like(text: str) -> str:
        """Best-effort extraction of JSON object from model output."""
        if not text:
            return ""
        # If fenced code blocks are present, strip them
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b != -1 and b > a:
            return text[a : b + 1]
        return text.strip()
