"""
Query rewriting / understanding utilities.

Goal:
- Produce two strings per user input:
  * retrieval_query: short, keyword-heavy, good for BM25 + embedding retrievers.
  * rerank_query: full intent/constraints (close to original) for cross-encoder rerank.

Design: pure rule-based (Phase 1, no LLM). Easily swappable later.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Set
from collections import Counter

try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import wordpunct_tokenize
    _NLTK_AVAILABLE = True
    _STEMMER = PorterStemmer()
except Exception:
    _NLTK_AVAILABLE = False
    _STEMMER = None

# Small, hand-curated vocabularies (extend as needed).
TECH_SKILLS = {
    "java",
    "python",
    "c++",
    "c#",
    ".net",
    "dotnet",
    "sql",
    "mysql",
    "postgres",
    "react",
    "node",
    "javascript",
    "typescript",
    "spring",
    "django",
    "flask",
    "aws",
    "azure",
    "gcp",
    "kubernetes",
    "docker",
    "html",
    "css",
    "jquery",
    "selenium",
    "angular",
    "vue",
    "ruby",
    "rails",
    "php",
    "laravel",
    "terraform",
    "ansible",
    "jenkins",
    "ci",
    "cd",
    "cicd",
    "ml",
    "ai",
    "data",
    "science",
}

SOFT_SKILLS = {
    "communication",
    "collaboration",
    "collaborate",
    "collaborative",
    "teamwork",
    "team",
    "stakeholder",
    "leadership",
    "analytical",
    "problem solving",
    "analytical thinking",
    "people management",
    "customer",
    "client",
    "communication",
    "negotiation",
    "presentation",
    "adaptability",
    "emotional",
    "conflict",
}

JOB_LEVEL_HINTS = {
    "entry": ["entry", "graduate", "junior"],
    "mid": ["mid", "mid-level", "midlevel"],
    "senior": ["senior", "sr", "lead"],
    "manager": ["manager", "management", "leadership"],
}

ROLE_HINTS = {
    "developer",
    "dev",
    "engineer",
    "manager",
    "analyst",
    "writer",
    "content",
    "designer",
    "consultant",
    "architect",
    "lead",
    "sales",
}

ROLE_PHRASES = [
    "java developer",
    "software engineer",
    "software developer",
    "content writer",
    "technical writer",
    "product manager",
    "project manager",
    "data analyst",
    "data scientist",
    "business analyst",
]

# Domain intent canonicalization: map abstract user phrases to catalog-relevant terms.
INTENT_CANONICAL_MAP = {
    "culture fit": ["personality", "behavioral", "values", "situational judgement"],
    "cultural fit": ["personality", "behavioral", "values", "situational judgement"],
    "leadership": ["leadership", "management", "executive"],
    "coo": ["executive", "leadership", "management"],
    "chief operating officer": ["executive", "leadership", "management"],
    "content writer": ["english comprehension", "verbal ability", "writing"],
    "seo": ["verbal reasoning", "english comprehension", "writing"],
    "culture fit": ["personality", "behavioral", "values", "situational judgement"],
    "collaborate": ["communication", "teamwork", "interpersonal communications"],
    "collaboration": ["communication", "teamwork", "interpersonal communications"],
    "communication": ["communication", "interpersonal communications"],
    "business team": ["communication", "teamwork", "interpersonal communications"],
}

STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "to",
    "with",
    "of",
    "a",
    "an",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "be",
    "that",
    "this",
    "these",
    "those",
    "as",
    "from",
    "we",
    "our",
    "their",
    "your",
    "i",
    "you",
    "they",
    "he",
    "she",
    "it",
    "was",
    "were",
    "will",
    "can",
    "could",
    "should",
    "would",
    "who",
    # Extra noise terms common in user JDs / requests that dilute retrieval queries
    "want",
    "hiring",
    "hire",
    "new",
    "role",
    "company",
    "compani",
    "my",
    "budget",
    "option",
    "options",
    "give",
    "some",
    "about",
    "each",
    "test",
    "tests",
}

SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "db": "database",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "comm": "communication",
}

MISSPELLINGS = {
    "pythn": "python",
    "javscript": "javascript",
    "dockr": "docker",
    "kubernets": "kubernetes",
}


@dataclass
class DurationConstraint:
    mode: str  # "MAX" or "TARGET"
    minutes: int


@dataclass
class ParsedConstraints:
    duration: Optional[DurationConstraint]
    job_levels: List[str]
    languages: List[str]
    experience: Optional[str]
    flags: Dict[str, Optional[bool]]  # remote/adaptive


@dataclass
class QueryRewrite:
    retrieval_query: str
    rerank_query: str
    intent: str  # TECH / BEHAVIORAL / MIXED / UNKNOWN
    must_have_skills: List[str]
    soft_skills: List[str]
    role_terms: List[str]
    negated_skills: List[str]
    constraints: ParsedConstraints
    llm_debug: Optional[dict] = None

    def to_dict(self):
        d = asdict(self)
        # dataclass for constraints nests another dataclass; fix for serialization
        if self.constraints and self.constraints.duration:
            d["constraints"]["duration"] = asdict(self.constraints.duration)
        if self.llm_debug is not None:
            d["llm_debug"] = self.llm_debug
        return d


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _lower(text: str) -> str:
    return _normalize(text.lower())


def parse_duration(text: str) -> Optional[DurationConstraint]:
    t = text.lower()
    if "about an hour" in t or "around an hour" in t or re.search(r"\ban hour\b", t):
        return DurationConstraint(mode="TARGET", minutes=60)
    if "half hour" in t:
        return DurationConstraint(mode="TARGET", minutes=30)
    # hours pattern e.g., 1.5 hours
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)", t)
    if m:
        minutes = int(round(float(m.group(1)) * 60))
        return DurationConstraint(mode="TARGET", minutes=minutes)
    # minutes pattern
    m = re.search(r"(\d{1,3})\s*(minute|min|minutes|mins)", t)
    if m:
        minutes = int(m.group(1))
        # MAX if “at most/within/under”; else TARGET
        if re.search(r"(at most|within|under|<=|less than)", t):
            return DurationConstraint(mode="MAX", minutes=minutes)
        return DurationConstraint(mode="TARGET", minutes=minutes)
    return None


def parse_flags(text: str) -> Dict[str, Optional[bool]]:
    t = text.lower()
    remote = True if re.search(r"\bremote\b", t) else None
    adaptive = True if re.search(r"\badaptive\b|\birt\b", t) else None
    return {"remote": remote, "adaptive": adaptive}


def parse_languages(text: str) -> List[str]:
    LANGS = {
        "english": "English",
        "spanish": "Spanish",
        "french": "French",
        "german": "German",
        "mandarin": "Mandarin",
    }
    langs = []
    t = text.lower()
    for key, val in LANGS.items():
        if key in t:
            langs.append(val)
    return langs


def parse_experience(text: str) -> Optional[str]:
    t = text.lower()
    m = re.search(r"(\d{1,2})(?:\s*-\s*(\d{1,2}))?\s*(year|years|yr|yrs)", t)
    if m:
        low = int(m.group(1))
        high = int(m.group(2)) if m.group(2) else low + 2
        return f"{low}-{high} years"
    if "fresher" in t or "0 years" in t or "entry-level" in t:
        return "0-2 years"
    return None


def parse_job_levels(text: str) -> List[str]:
    out = set()
    t = text.lower()
    for lvl, patterns in JOB_LEVEL_HINTS.items():
        for p in patterns:
            if re.search(rf"\b{re.escape(p)}\b", t):
                out.add(lvl.title())
    return sorted(out)


def tokenize(text: str) -> List[str]:
    def simple_stem(w: str) -> str:
        if w.endswith("ing") and len(w) > 4:
            return w[:-3]
        if w.endswith("s") and len(w) > 3:
            return w[:-1]
        return w

    if _NLTK_AVAILABLE:
        raw_tokens = wordpunct_tokenize(text.lower())
    else:
        raw_tokens = re.findall(r"[a-zA-Z0-9\+#\.]+", text.lower())

    toks = [MISSPELLINGS.get(t, SYNONYMS.get(t, t)) for t in raw_tokens]

    if _NLTK_AVAILABLE and _STEMMER is not None:
        toks = [_STEMMER.stem(t) for t in toks]
    else:
        toks = [simple_stem(t) for t in toks]

    return [t for t in toks if t and t not in STOPWORDS]


def extract_phrases(tokens: List[str], max_phrases: int = 5) -> List[str]:
    """Return a few informative bigrams/trigrams preserved as phrases."""
    phrases: List[str] = []
    n = len(tokens)
    SIGNAL = set(TECH_SKILLS) | set(ROLE_HINTS) | set(SOFT_SKILLS) | {
        "graduate",
        "junior",
        "entry",
        "senior",
        "manager",
        "leadership",
        "culture",
        "values",
        "personality",
        "behavior",
        "behaviour",
        "sales",
        "marketing",
    }
    for size in (3, 2):  # prefer trigrams, then bigrams
        for i in range(n - size + 1):
            gram = tokens[i : i + size]
            # require at least one signal token
            if not any(g in SIGNAL for g in gram):
                continue
            # skip if mostly stopwords/very short
            if all(t in STOPWORDS for t in gram):
                continue
            if sum(len(g) <= 2 for g in gram) >= 2:
                continue
            phrase = " ".join(gram)
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases[:max_phrases]


def extract_skills(tokens: List[str]) -> (Set[str], Set[str], Set[str]):
    toks_join = " ".join(tokens)
    must = set()
    soft = set()
    negated = set()
    for skill in TECH_SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", toks_join):
            must.add(skill)
    for s in SOFT_SKILLS:
        if re.search(rf"\b{re.escape(s)}\b", toks_join):
            soft.add(s)
    for i, tok in enumerate(tokens):
        if tok in TECH_SKILLS and i > 0 and tokens[i - 1] in {"no", "without", "exclude", "not"}:
            negated.add(tok)
            must.discard(tok)
    return must, soft, negated


def top_keywords(tokens: List[str], k: int = 15) -> List[str]:
    cnt = Counter(tokens)
    return [w for w, _ in cnt.most_common(k)]


def classify_intent(tokens: List[str]) -> str:
    tset = set(tokens)
    tech_hit = any(tok in TECH_SKILLS for tok in tset)
    behav_hit = any(tok in {"communication", "collaboration", "teamwork", "stakeholder", "leadership", "personality", "values", "culture", "cultural", "fit", "behavioral", "behavioural"} or tok.startswith("sales") for tok in tset)
    if tech_hit and behav_hit:
        return "MIXED"
    if tech_hit:
        return "TECH"
    if behav_hit:
        return "BEHAVIORAL"
    return "UNKNOWN"


LOCATION_TOKENS = {
    "china",
    "india",
    "usa",
    "uk",
    "europe",
    "us",
    "canada",
    "germany",
    "france",
}


def strip_locations(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in LOCATION_TOKENS]


def intent_canonical_terms(text_lower: str) -> List[str]:
    terms: List[str] = []
    for phrase, mapped in INTENT_CANONICAL_MAP.items():
        if phrase in text_lower:
            terms.extend(mapped)
    # dedupe preserve order
    out = []
    seen = set()
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_retrieval_query(
    role_terms: List[str],
    must_skills: List[str],
    soft_skills: List[str],
    constraints: ParsedConstraints,
    extra_terms: List[str],
    phrases: Optional[List[str]] = None,
    canonical_terms: Optional[List[str]] = None,
) -> str:
    parts = []
    if phrases:
        # Inject phrase tokens in two forms:
        # 1) underscore-joined to preserve as a single token (for BM25).
        # 2) original text to keep semantic signal for embeddings.
        for p in phrases:
            p_norm = p.strip()
            if not p_norm:
                continue
            parts.append(p_norm.replace(" ", "_"))
            parts.append(p_norm)
    parts.extend(role_terms)
    parts.extend(must_skills)
    # Downweight negated skills by excluding them (could prefix with "-skill" if BM25 supports).
    parts.extend(soft_skills)
    parts.extend(extra_terms)
    if canonical_terms:
        parts.extend(canonical_terms)
    if constraints.experience:
        parts.append(constraints.experience)
    if constraints.duration:
        if constraints.duration.mode == "MAX":
            parts.append("duration under")
        parts.append(f"{constraints.duration.minutes} minutes")
    if constraints.languages:
        parts.extend([l.lower() for l in constraints.languages])
    # keep order, drop dupes, and trim length
    def _is_stop(tok: str) -> bool:
        if not tok:
            return True
        base = tok.replace("_", " ")
        base_parts = base.split()
        # If all parts are stopwords, drop it.
        return all(p in STOPWORDS for p in base_parts)

    deduped = []
    seen = set()
    for p in parts:
        if p and p not in seen and not _is_stop(p):
            deduped.append(p)
            seen.add(p)
    query = _normalize(" ".join(deduped))
    # Trim to ~40 tokens to avoid bloating retriever input
    toks = query.split()
    if len(toks) > 40:
        query = " ".join(toks[:40])
    return query


def _boost_from_vocab(tokens: Set[str], vocab: Dict[str, List[str]], intent: str, max_terms: int = 5) -> List[str]:
    out: List[str] = []
    if not vocab:
        return out
    if intent in ("TECH", "MIXED") and "technical" in vocab:
        for w in vocab["technical"]:
            if w in tokens and w not in out:
                out.append(w)
            if len(out) >= max_terms:
                return out
    if intent in ("BEHAVIORAL", "MIXED") and "behavioral" in vocab:
        for w in vocab["behavioral"]:
            if w in tokens and w not in out:
                out.append(w)
            if len(out) >= max_terms:
                return out
    if "roles" in vocab:
        for w in vocab["roles"]:
            if w in tokens and w not in out:
                out.append(w)
            if len(out) >= max_terms:
                return out
    return out[:max_terms]


def _extract_json_text(raw):
    # NuExtractWrapper often returns dict with clean_output
    if isinstance(raw, dict):
        for k in ("clean_output", "output", "text"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # sometimes raw_output exists
        v = raw.get("raw_output")
        if isinstance(v, str) and v.strip():
            return v
        # if dict itself is already the parsed object
        if "retrieval_query" in raw or "rerank_query" in raw:
            return json.dumps(raw)
        return ""  # will fail loudly below

    # QwenRewriter might return str or dict
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    # anything else
    return str(raw)

def _coerce_json(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        raise ValueError("LLM returned empty output")

    # strip fences
    if "```" in s:
        s = s.replace("```json", "").replace("```", "").strip()

    # extract first JSON object
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        s = s[a:b+1]

    return json.loads(s)


def _rewrite_with_llm(raw_text: str, catalog_vocab: Optional[Dict[str, List[str]]], llm_extractor) -> Optional[QueryRewrite]:
    # Legacy placeholder will be overridden by the newer implementation below.
    return None


# --- Qwen-focused LLM rewrite (preferred) ---
LLM_SCHEMA = json.dumps(
    {
        "retrieval_query": "concise keyword-heavy string",
        "rerank_query": "full query text (keep intent/constraints)",
        "intent": "one of TECH, BEHAVIORAL, MIXED, UNKNOWN",
        "must_have_skills": ["string"],
        "soft_skills": ["string"],
        "role_terms": ["string"],
        "negated_skills": ["string"],
        "constraints": {
            "duration": {"mode": "one of MAX, TARGET, or null", "minutes": "integer or null"},
            "job_levels": ["string"],
            "languages": ["string"],
            "experience": "string or null",
            "flags": {"remote": "boolean or null", "adaptive": "boolean or null"},
        },
    },
    indent=2,
)


def _rewrite_with_llm(raw_text: str, catalog_vocab: Optional[Dict[str, List[str]]], llm_extractor) -> Optional[QueryRewrite]:
    """
    Preferred LLM rewrite using Qwen (or other local LLM). Returns None on failure so the
    caller can fall back to deterministic rewrite.
    """
    try:
        print(f"[LLM-RW] invoking {getattr(llm_extractor, 'model_name', type(llm_extractor))} on text='{raw_text[:80]}'")
        raw = llm_extractor.predict(text=raw_text, schema=LLM_SCHEMA, return_full=True)
        print(f"[LLM-RW] raw type={type(raw)} keys={list(raw.keys()) if isinstance(raw, dict) else 'n/a'}")
        if isinstance(raw, dict):
            print(
                f"[LLM-RW] raw_output head='{str(raw.get('raw_output',''))[:200]}' "
                f"clean_output head='{str(raw.get('clean_output',''))[:200]}'"
            )
        if "_extract_json_text" in globals():
            raw_json = _extract_json_text(raw)
        else:
            raw_json = raw.get("clean_output") if isinstance(raw, dict) else raw
        if not raw_json and isinstance(raw, dict) and raw.get("raw_output"):
            raw_json = raw.get("raw_output")
        print(f"[LLM-RW] raw_json head='{str(raw_json)[:120]}'")

        # If the model did not return JSON, coerce a minimal payload
        data = None
        if isinstance(raw_json, str) and "{" not in raw_json:
            data = {
                "retrieval_query": raw_json,
                "rerank_query": raw_text,
                "intent": None,
                "must_have_skills": [],
                "soft_skills": [],
                "role_terms": [],
                "negated_skills": [],
                "constraints": {"duration": None, "job_levels": [], "languages": [], "experience": None, "flags": {"remote": None, "adaptive": None}},
                "llm_coerced": True,
            }
        else:
            data = _coerce_json(raw_json) if "_coerce_json" in globals() else json.loads(raw_json)
        try:
            print(
                "[LLM-RW] parsed data:",
                {
                    "retrieval_query": str(data.get("retrieval_query"))[:80],
                    "rerank_query": str(data.get("rerank_query"))[:80],
                    "intent": data.get("intent"),
                    "must_have_skills": data.get("must_have_skills"),
                    "soft_skills": data.get("soft_skills"),
                    "role_terms": data.get("role_terms"),
                    "duration": data.get("constraints", {}).get("duration") if isinstance(data.get("constraints"), dict) else None,
                },
            )
        except Exception:
            pass

        dur = data.get("constraints", {}).get("duration") or {}
        duration_obj = None
        duration_error = None
        if dur.get("minutes") is not None:
            mode = dur.get("mode") or "TARGET"
            try:
                duration_obj = DurationConstraint(mode=mode, minutes=int(float(dur["minutes"])))
            except Exception as e:
                duration_error = f"duration_parse_error: {e}"
                duration_obj = None

        constraints = ParsedConstraints(
            duration=duration_obj,
            job_levels=data.get("constraints", {}).get("job_levels") or [],
            languages=data.get("constraints", {}).get("languages") or [],
            experience=data.get("constraints", {}).get("experience"),
            flags=data.get("constraints", {}).get("flags") or {"remote": None, "adaptive": None},
        )

        intent_raw = data.get("intent")
        allowed_intents = {"TECH", "BEHAVIORAL", "MIXED", "UNKNOWN"}
        intent_final = intent_raw if intent_raw in allowed_intents else None
        if intent_final is None:
            toks_src = data.get("retrieval_query") or raw_text
            toks = tokenize(_lower(toks_src))
            intent_final = classify_intent(toks)

        retrieval_q = data.get("retrieval_query") or raw_text
        rerank_q = data.get("rerank_query") or raw_text
        placeholder = False
        if retrieval_q.strip().lower() in {"string", ""} or rerank_q.strip().lower() in {"string", ""}:
            placeholder = True
        if intent_raw and "|" in str(intent_raw):
            placeholder = True

        rw = QueryRewrite(
            retrieval_query=retrieval_q,
            rerank_query=rerank_q,
            intent=intent_final or "UNKNOWN",
            must_have_skills=data.get("must_have_skills") or [],
            soft_skills=data.get("soft_skills") or [],
            role_terms=data.get("role_terms") or [],
            negated_skills=data.get("negated_skills") or [],
            constraints=constraints,
        )

        if isinstance(raw, dict):
            rw.llm_debug = {
                "prompt": raw.get("prompt"),
                "raw_output": raw.get("raw_output"),
                "clean_output": raw_json,
                "intent_raw": intent_raw,
                "model": getattr(llm_extractor, "model_name", "llm"),
            }
            if duration_error:
                rw.llm_debug["duration_error"] = duration_error
            if isinstance(data, dict) and data.get("llm_coerced"):
                rw.llm_debug["warning"] = "non_json_output_coerced"
        if placeholder:
            if rw.llm_debug is None:
                rw.llm_debug = {}
            rw.llm_debug["error"] = "placeholder_output"
            print("[LLM-RW] placeholder output, falling back")
            return None
        print(f"[LLM-RW] success model={rw.llm_debug.get('model') if rw.llm_debug else getattr(llm_extractor,'model_name','llm')} intent={intent_final}")
        return rw
    except Exception as e:
        print(f"[LLM-RW] error: {e}")
        return None
    """Try to rewrite via NuExtract (LLM) using a JSON schema; return None on failure."""
    schema ={
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
            },
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
    }

    try:
        raw = llm_extractor.predict(text=raw_text, schema=json.dumps(schema), return_full=True)
        print("LLM raw type:", type(raw))
        print("LLM raw keys:" , list(raw.keys()) if isinstance(raw, dict) else None)
        print("LLM raw_json head:", repr(raw_json[:80]))

        raw_json = _extract_json_text(raw)
        data = _coerce_json(raw_json)
        dur = data.get("constraints", {}).get("duration") or {}
        duration_obj = None
        duration_error = None
        if dur.get("minutes") is not None:
            mode = dur.get("mode") or "TARGET"
            try:
                minutes_val = float(dur["minutes"])
                duration_obj = DurationConstraint(mode=mode, minutes=int(minutes_val))
            except Exception as e:
                duration_obj = None
                duration_error = f"duration_parse_error: {e}"
        constraints = ParsedConstraints(
            duration=duration_obj,
            job_levels=data.get("constraints", {}).get("job_levels") or [],
            languages=data.get("constraints", {}).get("languages") or [],
            experience=data.get("constraints", {}).get("experience"),
            flags=data.get("constraints", {}).get("flags") or {"remote": None, "adaptive": None},
        )
        intent_raw = data.get("intent")
        allowed_intents = {"TECH", "BEHAVIORAL", "MIXED", "UNKNOWN"}
        intent_final = intent_raw if intent_raw in allowed_intents else None
        # fallback heuristic intent if LLM intent is missing/unrecognized
    
        retrieval_q = data.get("retrieval_query") or raw_text
        if intent_final is None or intent_final == "UNKNOWN":
            # Use LLM-derived retrieval query for a better hint
            toks_src = retrieval_q if isinstance(retrieval_q, str) else raw_text
            toks = tokenize(_lower(toks_src))
            intent_final = classify_intent(toks)

        rerank_q = data.get("rerank_query") or raw_text
        # Detect placeholder/hallucinated outputs; if placeholders found, treat as failure.
        placeholder = False
        if retrieval_q.strip().lower() in {"string", ""} or rerank_q.strip().lower() in {"string", ""}:
            placeholder = True
        if intent_raw == "TECH|BEHAVIORAL|MIXED|UNKNOWN":
            placeholder = True
        if dur.get("minutes") in ("int|null", "int", "", None) and duration_obj is None:
            placeholder = True

        rw = QueryRewrite(
            retrieval_query=retrieval_q,
            rerank_query=rerank_q,
            intent=intent_final or "UNKNOWN",
            must_have_skills=data.get("must_have_skills") or [],
            soft_skills=data.get("soft_skills") or [],
            role_terms=data.get("role_terms") or [],
            negated_skills=data.get("negated_skills") or [],
            constraints=constraints,
        )
        # attach LLM debug if available
        if isinstance(raw, dict):
            rw.llm_debug = {
                "prompt": raw.get("prompt"),
                "raw_output": raw.get("raw_output"),
                "clean_output": raw_json,
                "intent_raw": intent_raw,
                "model": getattr(llm_extractor, "model_name", "llm"),
            }
            if duration_error:
                rw.llm_debug["duration_error"] = duration_error
            if placeholder:
                rw.llm_debug["error"] = "placeholder_output"
        if placeholder:
            return None
        return rw
    except Exception as e:
        # Attach failure reason for debugging if caller wants it
        dummy = QueryRewrite(
            retrieval_query=raw_text,
            rerank_query=raw_text,
            intent="UNKNOWN",
            must_have_skills=[],
            soft_skills=[],
            role_terms=[],
            negated_skills=[],
            constraints=ParsedConstraints(duration=None, job_levels=[], languages=[], experience=None, flags={"remote": None, "adaptive": None}),
            llm_debug={"error": str(e)},
        )
        return dummy


def rewrite_query(raw_text: str, catalog_vocab: Optional[Dict[str, List[str]]] = None, llm_extractor=None) -> QueryRewrite:
    catalog_vocab = catalog_vocab or {}
    raw_clean = raw_text.strip()
    low = _lower(raw_text)
    # LLM-based rewrite first if provided
    llm_fail_debug = None
    if llm_extractor:
        print(f"[REWRITE] using LLM extractor {getattr(llm_extractor, 'model_name', type(llm_extractor))}")
        llm_rw = _rewrite_with_llm(raw_text, catalog_vocab, llm_extractor)
        if llm_rw and not (llm_rw.llm_debug and llm_rw.llm_debug.get("error")):
            # If LLM returned coerced/non-JSON output, fall back to deterministic rewrite for richer fields.
            warning = llm_rw.llm_debug.get("warning") if llm_rw.llm_debug else None
            if warning and "coerced" in warning:
                print("[REWRITE] LLM output was coerced/non-JSON; falling back to deterministic rewrite.")
            else:
                return llm_rw
        if llm_rw and llm_rw.llm_debug:
            llm_fail_debug = llm_rw.llm_debug
            print(f"[REWRITE] LLM rewrite failed, llm_debug={llm_fail_debug}")

    tokens = tokenize(low)
    tokens = strip_locations(tokens)

    duration = parse_duration(raw_text)
    flags = parse_flags(raw_text)
    languages = parse_languages(raw_text)
    experience = parse_experience(raw_text)
    job_levels = parse_job_levels(raw_text)
    constraints = ParsedConstraints(duration=duration, job_levels=job_levels, languages=languages, experience=experience, flags=flags)

    intent = classify_intent(tokens)
    must_skills, soft_sk, neg_skills = extract_skills(tokens)
    keywords = top_keywords(tokens, k=25)
    # Boost with catalog vocab matches, typed by intent.
    boost = _boost_from_vocab(set(tokens), catalog_vocab, intent, max_terms=5)

    # Role terms: prefer ROLE_HINTS present in tokens, then fall back to top keywords.
    role_terms: List[str] = []
    # Add matching role phrases if present.
    for phrase in ROLE_PHRASES:
        if phrase in low and phrase not in role_terms:
            role_terms.append(phrase)
    for tok in keywords:
        if tok in ROLE_HINTS and tok not in role_terms:
            role_terms.append(tok)
        if len(role_terms) >= 5:
            break
    if len(role_terms) < 3:  # backfill with keywords if needed
        for tok in keywords:
            if tok not in role_terms and tok not in STOPWORDS:
                role_terms.append(tok)
            if len(role_terms) >= 5:
                break
    phrases = extract_phrases(tokens, max_phrases=5)
    canonical_terms = intent_canonical_terms(low)
    retrieval_query = build_retrieval_query(
        role_terms,
        sorted(must_skills),
        sorted(soft_sk),
        constraints,
        boost,
        phrases=phrases,
        canonical_terms=canonical_terms,
    )
    rerank_query = raw_clean  # keep full context for reranker

    return QueryRewrite(
        retrieval_query=retrieval_query,
        rerank_query=rerank_query,
        intent=intent,
        must_have_skills=sorted(must_skills),
        soft_skills=sorted(soft_sk),
        role_terms=role_terms,
        negated_skills=sorted(neg_skills),
        constraints=constraints,
        llm_debug=llm_fail_debug,
    )


def build_catalog_vocab(catalog_texts: List[str], min_len: int = 5, max_terms: int = 200) -> Dict[str, List[str]]:
    """Lightweight vocab from catalog, bucketed by intent."""
    cnt_tech = Counter()
    cnt_behav = Counter()
    cnt_roles = Counter()
    for txt in catalog_texts:
        for tok in tokenize(txt.lower()):
            if len(tok) < min_len:
                continue
            if tok in TECH_SKILLS:
                cnt_tech[tok] += 1
            elif tok in SOFT_SKILLS or "behavior" in tok or "culture" in tok:
                cnt_behav[tok] += 1
            elif tok in ROLE_HINTS:
                cnt_roles[tok] += 1
    return {
        "technical": [w for w, _ in cnt_tech.most_common(max_terms // 3 or 1)],
        "behavioral": [w for w, _ in cnt_behav.most_common(max_terms // 3 or 1)],
        "roles": [w for w, _ in cnt_roles.most_common(max_terms // 3 or 1)],
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Rewrite a query into retrieval + rerank forms.")
    parser.add_argument("--query", required=True, help="Raw user query/JD text")
    parser.add_argument("--catalog", help="Optional catalog JSONL to build vocab (uses doc_text or name/description)")
    args = parser.parse_args()

    vocab = {}
    if args.catalog:
        import pandas as pd

        df = pd.read_json(args.catalog, lines=True)
        if "doc_text" in df.columns:
            texts = df["doc_text"].astype(str).tolist()
        else:
            texts = (df.get("name", "").astype(str) + " " + df.get("description", "").astype(str)).tolist()
        vocab = build_catalog_vocab(texts)

    rewrite = rewrite_query(args.query, vocab)
    json.dump(rewrite.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
