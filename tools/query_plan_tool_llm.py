from __future__ import annotations

"""
LLM-based QueryPlan builder using Gemini + LangChain structured output.
Falls back to deterministic rewrite if LLM parsing fails.
"""

import os
from typing import Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from schemas.query_plan import QueryPlan
from tools.query_plan_tool import build_query_plan as deterministic_plan

SYSTEM_PROMPT = """You are a retrieval planner for assessment recommendations.
Extract intent, role, skills, duration, language. Produce BM25/vec queries (keyword-heavy)
and a rerank query (full original). Keep to the schema exactly.
"""

TEMPLATE = """{system}
User query:
{query}

Return ONLY valid JSON for the QueryPlan model.
"""


def build_query_plan_llm(raw_text: str, vocab: Optional[Dict] = None, model_name: str = "gemini-pro") -> QueryPlan:
    try:
        parser = PydanticOutputParser(pydantic_object=QueryPlan)
        prompt = PromptTemplate(
            template=TEMPLATE,
            input_variables=["query"],
            partial_variables={"system": SYSTEM_PROMPT},
        ).partial(format_instructions=parser.get_format_instructions())

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_output_tokens=512,
            convert_system_message_to_human=True,
        )
        chain = prompt | llm | parser
        return chain.invoke({"query": raw_text})
    except Exception as e:
        # Fallback to deterministic rewriter
        return deterministic_plan(raw_text, vocab=vocab)
