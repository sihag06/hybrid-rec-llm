from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import List, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=1)
def _load_model():
    # Use a stronger model for better structured extraction; falls back to tiny if unavailable.
    model_name = "numind/NuExtract"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
    model.eval()
    return model, tokenizer


class NuExtractWrapper:
    """Lightweight wrapper around NuExtract-tiny for structured extraction."""

    def __init__(self, device: Optional[str] = None, default_examples: Optional[List[str]] = None) -> None:
        self.model, self.tokenizer = _load_model()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.default_examples = default_examples or []
        # Keep prompt minimal: NuExtract expects just template + examples + text.
        self.system_prompt = ""

    def predict(
        self,
        text: str,
        schema: str,
        examples: Optional[List[str]] = None,
        max_length: int = 4000,
        system_prompt: Optional[str] = None,
        return_full: bool = False,
    ):
        """
        Return the raw JSON string extracted per schema.
        `schema` should be a JSON string describing expected keys.
        `examples` is an optional list of JSON strings.
        """
        examples = examples or self.default_examples
        schema_fmt = json.dumps(json.loads(schema), indent=4)
        sys_prompt = system_prompt or self.system_prompt
        input_llm = "<|input|>\n"
        if sys_prompt:
            input_llm += sys_prompt + "\n"
        input_llm += "### Template:\n" + schema_fmt + "\n"
        for ex in examples:
            if ex:
                input_llm += "### Example:\n" + json.dumps(json.loads(ex), indent=4) + "\n"
        input_llm += "### Text:\n" + text + "\n<|output|>\n"

        inputs = self.tokenizer(
            input_llm,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                top_p=0.9,
            )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "<|output|>" in output:
            output = output.split("<|output|>", 1)[1]
        if "<|end-output|>" in output:
            output = output.split("<|end-output|>", 1)[0]
        cleaned = output.strip()
        # Clean common placeholder/enums
        cleaned = re.sub(r"\\|[A-Z]+", "", cleaned)
        cleaned = cleaned.replace("TECH|BEHAVIORAL|MIXED|UNKNOWN", "UNKNOWN")
        cleaned = cleaned.replace("MAX|TARGET|null", "null")
        cleaned = cleaned.replace("int|null", "null")
        cleaned = cleaned.replace("string", "\"\"")
        if return_full:
            return {"prompt": input_llm, "raw_output": output, "clean_output": cleaned}
        return cleaned


def default_query_rewrite_examples() -> List[str]:
    """Few-shot examples for assessment query rewriting."""
    return [
        json.dumps(
            {
                "retrieval_query": "java developer assessment core java collaboration communication 40 minutes",
                "rerank_query": "Hiring Java dev who can collaborate with business teams. 40 minutes.",
                "intent": "MIXED",
                "must_have_skills": ["java"],
                "soft_skills": ["communication", "collaboration"],
                "role_terms": ["java developer"],
                "negated_skills": [],
                "constraints": {
                    "duration": {"mode": "TARGET", "minutes": 40},
                    "job_levels": [],
                    "languages": [],
                    "experience": None,
                    "flags": {"remote": None, "adaptive": None},
                },
            }
        ),
        json.dumps(
            {
                "retrieval_query": "culture fit leadership personality situational judgement executive assessment 60 minutes",
                "rerank_query": "Find a 1 hour culture fit assessment for a COO",
                "intent": "BEHAVIORAL",
                "must_have_skills": [],
                "soft_skills": ["leadership", "personality"],
                "role_terms": ["coo", "executive"],
                "negated_skills": [],
                "constraints": {
                    "duration": {"mode": "TARGET", "minutes": 60},
                    "job_levels": ["manager"],
                    "languages": [],
                    "experience": None,
                    "flags": {"remote": None, "adaptive": None},
                },
            }
        ),
    ]
