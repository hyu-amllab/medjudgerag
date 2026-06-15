#!/usr/bin/env python3
"""
evaluate.py — Benchmark evaluation for medical MCQ.

Modes:
  parametric   — Q + O only, no documents (MedRAG CoT prompt)
  vanilla_rag  — Q + O + D with MedRAG prompt
  implicit     — Q + O + D with structured verdict/decision prompt (no KG generation)
  explicit     — Q + O + D with KG construction + option-wise reasoning

Legacy aliases are also accepted: naive_rag=vanilla_rag, analysis_only=implicit, stage2=explicit.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
        --model_path outputs/merged_mistral_kgw00 \
        --mode explicit --benchmark medqa --tag mistral_explicit --vllm
"""

import io
import json
import os
import random
import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


ALL_BENCHMARKS = ["medqa", "medmcqa"]

MODE_ALIASES = {
    "vanilla_rag": "naive_rag",
    "implicit": "analysis_only",
    "explicit": "stage2",
    "naive_rag": "naive_rag",
    "analysis_only": "analysis_only",
    "stage2": "stage2",
    "parametric": "parametric",
}


# ── Prompts ──────────────────────────────────────────────────────

# --- Parametric (CoT, no documents) — MedRAG general_cot ---
PARAMETRIC_SYSTEM = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/D}}. Your responses will be used for research purposes only, so please have a definite answer.'''

PARAMETRIC_USER = '''
Here is the question:
{question}

Here are the potential choices:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Please think step-by-step and generate your output in json:
'''

# --- Naive RAG — MedRAG general_medrag ---
NAIVE_RAG_SYSTEM = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/D}}. Your responses will be used for research purposes only, so please have a definite answer.'''

NAIVE_RAG_USER = '''
Here are the relevant documents:
{documents}

Here is the question:
{question}

Here are the potential choices:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Please think step-by-step and generate your output in json:
'''

# --- Implicit: MedJudgeRAG analysis prompt without KG generation ---
ANALYSIS_ONLY_SYSTEM = "You are a biomedical information extractor and medical reasoner."

ANALYSIS_ONLY_USER = '''\
====================
[Step 2: Analysis Rules]
====================
This mode skips Step 1 KG construction.
Assume no KG was generated.

Then, for each option (A/B/C/D), first find relevant references, then judge:
   - Doc: list of integer document IDs relevant to this option ([] if none)
   - KG Entities: list of relevant entity names from the KG you built ([] if none), e.g. ["Entity_X"] or ["Entity_X", "Entity_Y"]
   - KG Relations: list of relevant Relation IDs from the KG you built ([] if none), e.g. [Ri] or [Ri, Rj]
   - Evidence: explanation referencing ONLY the Doc/KG Entities/KG Relations listed above. If all three are [], Evidence must be "No relevant evidence found." — do NOT use medical knowledge here.
   - Verdict: exactly one of SUPPORTED, CONTRADICTED, or INSUFFICIENT

-Verdicts (relative to the Question)-
- SUPPORTED: Doc, KG Entities, or KG Relations provide evidence that this option correctly answers the Question. Requires at least one reference in Doc, KG Entities, or KG Relations.
- CONTRADICTED: Doc, KG Entities, or KG Relations provide evidence against this option as the answer. Requires at least one reference in Doc, KG Entities, or KG Relations.
- INSUFFICIENT: Doc, KG Entities, and KG Relations lack relevant information to judge this option. If Doc=[], KG Entities=[], and KG Relations=[], verdict MUST be INSUFFICIENT regardless of your medical knowledge.

-Decision-
Decision must be consistent with verdicts:
- "grounded" if at least one option is SUPPORTED
- "elimination" if no option is SUPPORTED but at least one is CONTRADICTED
- "parametric" only when ALL options are INSUFFICIENT
Summary: reasoning that leads to the final answer. The content depends on Decision:
- grounded: synthesize ONLY the SUPPORTED evidence to justify the answer. No parametric knowledge.
- elimination: state which options are ruled out by CONTRADICTED evidence, then use medical knowledge to choose among the remaining INSUFFICIENT options.
- parametric: use medical knowledge to reason through all options and justify the answer.

-Analysis output format (follow exactly)-
[A] Doc: [1, 3]
KG Entities: ["Entity_X"]
KG Relations: [Ri, Rj]
Evidence: Document [1] and KG relation Ri indicate that ...
Verdict: SUPPORTED

[B] Doc: []
KG Entities: ["Entity_Y"]
KG Relations: [Rk]
Evidence: KG relation Rk indicates that ...
Verdict: CONTRADICTED

[C] Doc: []
KG Entities: []
KG Relations: []
Evidence: No relevant evidence found.
Verdict: INSUFFICIENT

[D] Doc: []
KG Entities: []
KG Relations: []
Evidence: No relevant evidence found.
Verdict: INSUFFICIENT

Decision: grounded|elimination|parametric
Summary: ...
Answer_choice: A|B|C|D

-Order Constraint-
Always output in this order: option analysis → Decision/Summary/Answer_choice

Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Documents:
{documents}
Output:
'''

# --- Explicit: compact KG construction + analysis ---
STAGE2_SYSTEM = "You are a biomedical information extractor and medical reasoner."

STAGE2_USER = """\
-Task Overview-
Do two steps in order:
1) Build a question-focused Knowledge Graph from the Documents.
2) After the KG, analyze each option and choose the final answer.

====================
[Step 1: KG Rules]
====================
-Role-
You are a biomedical knowledge graph constructor. Extract a question-focused Dynamic Knowledge Graph (KG) from retrieved documents.

-Goal-
Extract entities and relations that help DISCRIMINATE among the given options.

-Entity constraints-
- Entity types: Activities & Behaviors | Anatomy | Chemicals & Drugs | Concepts & Ideas | Devices | Disorders | Genes & Molecular Sequences | Geographic Areas | Living Beings | Objects | Occupations | Organizations | Phenomena | Physiology | Procedures
- Entity Name MUST appear verbatim in the Documents (abbreviation ↔ full-form normalization is allowed ONLY when BOTH forms appear in the Documents).
- Prefer higher-level entities over their enumerated sub-items; if a collective concept already captures individual items, do not list them separately.
- Do NOT create entities directly from the answer Options. Entities must originate from the Documents. (Option text may help you recognize relevant concepts in Documents, but the Options themselves are NOT a source.)
- Entity Description: paraphrase from Documents only; no external knowledge.
- Entity Evidence: document id(s) only, e.g. [1] or [1, 3].

-Relation constraints-
- Relation types: part_of | located_in | connected_to | adjacent_to | performs | uses | affects | causes | result_of | indicates | measures | diagnoses | manifestation_of | precedes | co_occurs_with
- If a document explicitly states a negative relation and that negation is important for discriminating options, encode the negation in the Relation Description by starting it with "[NEGATED]". Use "[NEGATED]" only when the source document contains explicit negation cues (e.g., "no", "not", "did not", "without", "failed to", "absence of").
- Use only entity names from the extracted Entities list.
- Extract a relation only when explicitly or conservatively supported by Documents.
- Prioritize relations that directly discriminate among the given Options.
- Relation Source and Target MUST be copied exactly from the Entity Name field (character-for-character). If a needed endpoint is not already in the Entities list, prefer omitting the relation rather than adding a new low-relevance entity.
- Use "indicates" only when the document explicitly links X as a diagnostic clue or criterion for Y. Otherwise prefer a more conservative relation type.
- Use co_occurs_with only when no mechanistic relation applies. Prefer causal/mechanistic types (causes, result_of, indicates, manifestation_of) over co_occurs_with.
- When a document describes a causal or temporal chain (A → B → C), preserve intermediate steps as separate relations rather than collapsing into a single A → C link.
- Relation Evidence: document id(s) only, e.g. [1] or [1, 2].

-Global Rules-
- Do NOT answer the Question or choose an option.
- Use ONLY document-grounded facts; no external knowledge.
- Keep the graph concise: high relevance over exhaustive coverage.
- Not all documents are equally relevant. Focus on documents that directly address the question's core claim; skip tangential ones.
- Prioritize pathognomonic findings, key differentiating features, and diagnostic criteria that help distinguish among the given Options.
- All Evidence must cite numbered documents [1]-[N] only. Never use [Question] or [Options] as an evidence source.
- If no document contains information relevant to the Question, output an empty graph.

-Output format (strict format, no JSON, no markdown)-
Entities:
("Entity", <Name>, <Type>, <Description>, <Evidence>)

Relations:
R1: ("Relation", <Source>, <RelationType>, <Target>, <Description>, <Evidence>)
R2: ...

If no document is relevant, output only the headers with no entries:
Entities:

Relations:

######################
-Example-
######################
Question: A 12-year-old girl presents to her primary care physician with left knee pain for the past 6 weeks. She recently joined the field hockey team at her school. The pain is the most severe when she is running up and down the stairs at the school stadium. The pain decreases when she goes home and rests after practice. She additionally admits to tripping and landing on her left knee 5 days ago. Physical exam shows a knee with a healing abrasion over the left patella. The tibial tuberosity is tender to palpation. A radiograph of the knee is presented in figure A. Which of the following is the most likely diagnosis?
Options:
A. Osgood-Schlatter disease
B. Patellofemoral pain syndrome
C. Pes anserine bursitis
D. Tibial plateau fracture
Documents:
[1] An active 13-year-old boy has anterior knee pain. Diagnosis? The most common 1° malignant tumor of bone. Pseudogout. Polymyalgia rheumatica. Osgood-Schlatter disease. Distal radius (Colles' fracture). Avascular necrosis.
######################
Output:
Entities:
("Entity", "Osgood-Schlatter disease", "Disorders", "Anterior knee pain condition in active adolescents.", "[1]")
("Entity", "anterior knee pain", "Phenomena", "Symptom linked to Osgood-Schlatter disease.", "[1]")

Relations:
R1: ("Relation", "anterior knee pain", "indicates", "Osgood-Schlatter disease", "Anterior knee pain in an active adolescent indicates Osgood-Schlatter disease.", "[1]")
######################

====================
[Step 2: Analysis Rules]
====================
After finishing the KG, output this delimiter exactly:
<ANALYSIS>

Do not stop after KG. You must continue and output the full <ANALYSIS> section.

Then, for each option (A/B/C/D), first find relevant references, then judge:
   - Doc: list of integer document IDs relevant to this option ([] if none)
   - KG Entities: list of relevant entity names from the KG you built ([] if none), e.g. ["Entity_X"] or ["Entity_X", "Entity_Y"]
   - KG Relations: list of relevant Relation IDs from the KG you built ([] if none), e.g. [Ri] or [Ri, Rj]
   - Evidence: explanation referencing ONLY the Doc/KG Entities/KG Relations listed above. If all three are [], Evidence must be "No relevant evidence found." — do NOT use medical knowledge here.
   - Verdict: exactly one of SUPPORTED, CONTRADICTED, or INSUFFICIENT

-Verdicts (relative to the Question)-
- SUPPORTED: Doc, KG Entities, or KG Relations provide evidence that this option correctly answers the Question. Requires at least one reference in Doc, KG Entities, or KG Relations.
- CONTRADICTED: Doc, KG Entities, or KG Relations provide evidence against this option as the answer. Requires at least one reference in Doc, KG Entities, or KG Relations.
- INSUFFICIENT: Doc, KG Entities, and KG Relations lack relevant information to judge this option. If Doc=[], KG Entities=[], and KG Relations=[], verdict MUST be INSUFFICIENT regardless of your medical knowledge.

-Decision-
Decision must be consistent with verdicts:
- "grounded" if at least one option is SUPPORTED
- "elimination" if no option is SUPPORTED but at least one is CONTRADICTED
- "parametric" only when ALL options are INSUFFICIENT
Summary: reasoning that leads to the final answer. The content depends on Decision:
- grounded: synthesize ONLY the SUPPORTED evidence to justify the answer. No parametric knowledge.
- elimination: state which options are ruled out by CONTRADICTED evidence, then use medical knowledge to choose among the remaining INSUFFICIENT options.
- parametric: use medical knowledge to reason through all options and justify the answer.

-Analysis output format (follow exactly)-
[A] Doc: [1, 3]
KG Entities: ["Entity_X"]
KG Relations: [Ri, Rj]
Evidence: Document [1] and KG relation Ri indicate that ...
Verdict: SUPPORTED

[B] Doc: []
KG Entities: ["Entity_Y"]
KG Relations: [Rk]
Evidence: KG relation Rk indicates that ...
Verdict: CONTRADICTED

[C] Doc: []
KG Entities: []
KG Relations: []
Evidence: No relevant evidence found.
Verdict: INSUFFICIENT

[D] Doc: []
KG Entities: []
KG Relations: []
Evidence: No relevant evidence found.
Verdict: INSUFFICIENT

Decision: grounded|elimination|parametric
Summary: ...
Answer_choice: A|B|C|D

-Order Constraint-
Always output in this order: KG block → <ANALYSIS> → option analysis → Decision/Summary/Answer_choice

Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Documents:
{documents}
Output:
"""

MAX_TOKENS_BY_MODE = {
    "parametric": 1024,
    "naive_rag": 2048,
    "analysis_only": 3072,
    "stage2": 8192,
}

STRUCTURED_MODES = ("stage2", "analysis_only")


# ── CLI ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Standalone full model path")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["parametric", "vanilla_rag", "implicit", "explicit",
                                 "naive_rag", "analysis_only", "stage2"])
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["medqa", "medmcqa", "all"])
    parser.add_argument("--data_path", type=str, default=None,
                        help="Optional jsonl path to evaluate directly. "
                             "When set, this file is used instead of data/<benchmark>_eval_retrieval_ids.jsonl.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--flush_every", type=int, default=20)
    parser.add_argument("--print_each", action="store_true",
                        help="Print per-sample prediction summary")
    parser.add_argument("--print_only_wrong", action="store_true",
                        help="When --print_each is set, print only wrong samples")
    parser.add_argument("--vllm", action="store_true",
                        help="Use vLLM for batch inference")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.90)
    return parser.parse_args()


# ── Data ─────────────────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def resolve_data_path(benchmark):
    path = Path("data") / f"{benchmark}_eval_retrieval_ids.jsonl"
    if not path.exists():
        legacy = Path("outputs") / f"test_context_{benchmark}.jsonl"
        if legacy.exists():
            return legacy
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


# ── Model loading ────────────────────────────────────────────────
def load_model(model_path, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def load_model_vllm(model_path, gpu_memory_utilization=0.90):
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        seed=42,
    )
    return tokenizer, llm


# ── Prompt construction ──────────────────────────────────────────
def build_prompt(mode, question, passages, options):
    opt_a = options.get("A", "")
    opt_b = options.get("B", "")
    opt_c = options.get("C", "")
    opt_d = options.get("D", "")

    if mode == "parametric":
        return PARAMETRIC_SYSTEM, PARAMETRIC_USER.format(
            question=question, option_a=opt_a, option_b=opt_b,
            option_c=opt_c, option_d=opt_d,
        )
    elif mode == "naive_rag":
        documents = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        return NAIVE_RAG_SYSTEM, NAIVE_RAG_USER.format(
            question=question, option_a=opt_a, option_b=opt_b,
            option_c=opt_c, option_d=opt_d, documents=documents,
        )
    elif mode == "analysis_only":
        documents = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        return ANALYSIS_ONLY_SYSTEM, ANALYSIS_ONLY_USER.format(
            question=question, option_a=opt_a, option_b=opt_b,
            option_c=opt_c, option_d=opt_d, documents=documents,
        )
    elif mode == "stage2":
        documents = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        return STAGE2_SYSTEM, STAGE2_USER.format(
            question=question, option_a=opt_a, option_b=opt_b,
            option_c=opt_c, option_d=opt_d, documents=documents,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")



# ── Answer extraction ────────────────────────────────────────────
def extract_answer(text):
    m = re.search(r'Answer_choice\s*:\s*([A-Da-d])', text)
    if m:
        return m.group(1).upper()

    for pat in [
        r'"answer_choice"\s*:\s*"\s*([A-Da-d])(?:\b|\s*[\.)-])',
        r'"final_choice"\s*:\s*"\s*([A-Da-d])(?:\b|\s*[\.)-])',
        r'(?:answer_choice|final_choice)\s*:\s*"?\s*([A-Da-d])(?:\b|\s*[\.)-])',
        r'"(?:answer_choice|final_choice)"\s*:\s*"\s*(?:option|choice)\s*([A-Da-d])(?:\b|\s*[\.)-])',
        r'(?:answer_choice|final_choice)\s*:\s*"?\s*(?:option|choice)\s*([A-Da-d])(?:\b|\s*[\.)-])',
        r'"(?:answer_choice|final_choice)"\s*:\s*"[^"\n]*\(([A-Da-d])\)\s*"',
        r'(?:answer_choice|final_choice)\s*:\s*"?[^"\n]*\(([A-Da-d])\)\s*"?',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = re.search(r'\bAnswer\s*:\s*([A-Da-d])(?:\b|\s*[\.)-])', text)
    if m:
        return m.group(1).upper()

    m = re.search(
        r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?([A-Da-d])(?:\b|\s*[\.)-])',
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    m = re.search(
        r'\b(?:i\s+)?(?:choose|chose|pick|picked|select|selected)\s+(?:option\s+)?([A-Da-d])(?:\b|\s*[\.)-])',
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    m = re.findall(r'(?:^|\n)\s*([A-Da-d])\s*$', text)
    if m:
        return m[-1].upper()

    return "other"


# ── Structured output parsing ────────────────────────────────────
def parse_verdicts(text):
    verdicts = {}
    opts = ["A", "B", "C", "D"]
    for i, opt in enumerate(opts):
        start_m = re.search(rf'\[{opt}\]', text)
        if not start_m:
            continue
        start = start_m.start()
        if i + 1 < len(opts):
            end_m = re.search(rf'\[{opts[i+1]}\]', text[start + 1:])
            end = start + 1 + end_m.start() if end_m else len(text)
        else:
            end_m = re.search(r'\nDecision\s*:', text[start:])
            end = start + end_m.start() if end_m else len(text)
        block = text[start:end]
        m = re.search(r'Verdict\s*:\s*(SUPPORTED|CONTRADICTED|INSUFFICIENT)', block)
        if m:
            verdicts[opt] = m.group(1)
    return verdicts


def parse_decision(text):
    m = re.search(r'Decision\s*:\s*(grounded|elimination|parametric)', text)
    return m.group(1) if m else None


def check_kg_generated(text):
    analysis_pos = text.find("<ANALYSIS>")
    if analysis_pos <= 0:
        return 0, 0
    kg_section = text[:analysis_pos]
    n_entities = len(re.findall(r'^\s*\("Entity"', kg_section, re.MULTILINE))
    n_relations = len(re.findall(r'^\s*\("Relation"', kg_section, re.MULTILINE))
    return n_entities, n_relations


# ── Generation (HuggingFace) ────────────────────────────────────
def generate(tokenizer, model, system, user, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    max_ctx = (
        getattr(model.config, "max_position_embeddings", None)
        or tokenizer.model_max_length
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_ctx
    )
    input_len = inputs.input_ids.shape[1]
    room = max_ctx - input_len - 32
    if room < 256:
        print(f"  WARNING: only {room} tokens of context room (input_len={input_len})")
    allowed = min(max_new_tokens, max(0, room))
    if allowed <= 0:
        return ""
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=allowed,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


# ── Stratified sampling ──────────────────────────────────────────
_UUID_V4_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def infer_subject(qid, benchmark=None):
    if not isinstance(qid, str) or not qid:
        return benchmark or "unknown"
    if qid.startswith("medqa_"):
        return "medqa"
    if qid.startswith("medmcqa_"):
        return "medmcqa"
    if "-" in qid:
        left, right = qid.rsplit("-", 1)
        if right.isdigit() and left:
            return left
    if _UUID_V4_RE.match(qid):
        return benchmark or "medmcqa"
    return benchmark or qid


def stratified_sample(data, n, benchmark=None, seed=42):
    if n >= len(data):
        return data
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in data:
        groups[infer_subject(item["id"], benchmark)].append(item)

    sorted_subjects = sorted(groups.keys())
    total = len(data)
    alloc = {}
    remainder = []
    allocated = 0
    for subj in sorted_subjects:
        count = len(groups[subj])
        base_alloc = int(n * count / total)
        frac = (n * count / total) - base_alloc
        alloc[subj] = base_alloc
        allocated += base_alloc
        remainder.append((frac, subj))
    remainder.sort(key=lambda x: -x[0])
    for _, subj in remainder:
        if allocated >= n:
            break
        alloc[subj] += 1
        allocated += 1

    sampled = []
    for subj in sorted_subjects:
        pool = groups[subj]
        rng.shuffle(pool)
        sampled.extend(pool[:alloc[subj]])
    rng.shuffle(sampled)
    return sampled


# ── Evaluation (shared logic) ────────────────────────────────────
def process_result(entry, raw, mode):
    """Process a single generation result into a result dict + stats."""
    label = entry["label"]
    pred = extract_answer(raw)

    verdicts = {}
    decision = None
    n_ent, n_rel = 0, 0
    has_analysis = False

    if mode in STRUCTURED_MODES:
        verdicts = parse_verdicts(raw)
        decision = parse_decision(raw)
    if mode == "stage2":
        n_ent, n_rel = check_kg_generated(raw)
        has_analysis = "<ANALYSIS>" in raw

    result = {
        "id": entry["id"],
        "label": label,
        "prediction": pred,
        "correct": pred == label,
        "raw_generation": raw,
    }
    if mode in STRUCTURED_MODES:
        result["decision"] = decision
        result["verdicts"] = verdicts
    if mode == "stage2":
        result["kg_entities"] = n_ent
        result["kg_relations"] = n_rel

    return result, has_analysis, n_ent, n_rel


def aggregate_stats(results, mode):
    """Compute aggregate stats from a list of result dicts."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    pred_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}
    decision_counts = {"grounded": 0, "elimination": 0, "parametric": 0, "none": 0}
    route_counts = {}
    fallback_reason_counts = {}
    has_analysis = 0
    has_kg = 0
    kg_ent, kg_rel = [], []

    for r in results:
        pred_counts[r["prediction"] if r["prediction"] in pred_counts else "other"] += 1
        if mode in STRUCTURED_MODES:
            decision_counts[r.get("decision") or "none"] += 1
        route = r.get("route")
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
        fallback_reason = r.get("fallback_reason")
        if fallback_reason:
            fallback_reason_counts[fallback_reason] = fallback_reason_counts.get(fallback_reason, 0) + 1
        if mode == "stage2":
            ne = r.get("kg_entities", 0)
            nr = r.get("kg_relations", 0)
            kg_ent.append(ne)
            kg_rel.append(nr)
            if ne > 0 or nr > 0:
                has_kg += 1
            if "<ANALYSIS>" in r.get("raw_generation", ""):
                has_analysis += 1

    acc = correct / max(total, 1)
    stats = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "pred_counts": pred_counts,
        "decision_counts": decision_counts,
        "route_counts": route_counts,
        "fallback_reason_counts": fallback_reason_counts,
        "has_analysis": has_analysis,
        "has_kg": has_kg,
        "empty_kg": sum(1 for e, r in zip(kg_ent, kg_rel) if e == 0 and r == 0) if kg_ent else 0,
        "avg_entities": round(np.mean(kg_ent), 1) if kg_ent else 0,
        "avg_relations": round(np.mean(kg_rel), 1) if kg_rel else 0,
    }
    return acc, stats


def _render_prompt(tokenizer, mode, entry):
    system, user = build_prompt(
        mode, entry["question"], entry.get("retrieved_passages", []), entry["choices"]
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── vLLM batch evaluation ────────────────────────────────────────
def run_eval_vllm(data, tokenizer, llm, mode, out_path, max_new_tokens,
                  benchmark=None, print_each=False, print_only_wrong=False):
    from vllm import SamplingParams

    print(f"  Building {len(data)} prompts...")
    prompts = []
    for entry in data:
        system, user = build_prompt(
            mode, entry["question"],
            entry.get("retrieved_passages", []), entry["choices"],
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    params = SamplingParams(temperature=0, max_tokens=max_new_tokens, seed=42)
    print(f"  Running vLLM batch inference ({len(prompts)} prompts, "
          f"max_tokens={max_new_tokens})...")
    outputs = llm.generate(prompts, params)

    results = []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (entry, output) in enumerate(zip(data, outputs), start=1):
            raw = output.outputs[0].text.strip()
            result, _, _, _ = process_result(entry, raw, mode)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if print_each:
                is_wrong = not result["correct"]
                if (not print_only_wrong) or is_wrong:
                    status = "OK" if result["correct"] else "WRONG"
                    msg = (
                        f"[{benchmark or 'custom'}|{mode}] "
                        f"{i}/{len(data)} id={result['id']} "
                        f"gold={result['label']} pred={result['prediction']} {status}"
                    )
                    if mode in STRUCTURED_MODES:
                        msg += f" decision={result.get('decision') or 'none'}"
                    if mode == "stage2":
                        msg += (
                            f" kgE={result.get('kg_entities', 0)}"
                            f" kgR={result.get('kg_relations', 0)}"
                        )
                    print(msg)

    return aggregate_stats(results, mode)



# ── HuggingFace sequential evaluation ───────────────────────────
def run_eval(data, tokenizer, model, mode, out_path, max_new_tokens,
             flush_every, benchmark=None, print_each=False, print_only_wrong=False):
    results = []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(
            tqdm(data, desc=f"eval [{benchmark or 'custom'}|{mode}]"),
            start=1,
        ):
            system, user = build_prompt(
                mode, entry["question"],
                entry.get("retrieved_passages", []), entry["choices"],
            )
            raw = generate(tokenizer, model, system, user, max_new_tokens)
            result, _, _, _ = process_result(entry, raw, mode)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if len(results) % max(1, flush_every) == 0:
                f.flush()

            if print_each:
                is_wrong = not result["correct"]
                if (not print_only_wrong) or is_wrong:
                    status = "OK" if result["correct"] else "WRONG"
                    msg = (
                        f"[{benchmark or 'custom'}|{mode}] "
                        f"{i}/{len(data)} id={result['id']} "
                        f"gold={result['label']} pred={result['prediction']} {status}"
                    )
                    if mode in STRUCTURED_MODES:
                        msg += f" decision={result.get('decision') or 'none'}"
                    if mode == "stage2":
                        msg += (
                            f" kgE={result.get('kg_entities', 0)}"
                            f" kgR={result.get('kg_relations', 0)}"
                        )
                    print(msg)

    return aggregate_stats(results, mode)



# ── Main ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    original_mode = args.mode
    args.mode = MODE_ALIASES[args.mode]
    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_path = args.model_path
    tag = args.tag
    max_new_tokens = args.max_new_tokens or MAX_TOKENS_BY_MODE[args.mode]

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_vllm = args.vllm
    print(f"Loading model: {model_path} ({'vLLM' if use_vllm else 'HuggingFace'})")
    if use_vllm:
        tokenizer, llm = load_model_vllm(model_path, args.vllm_gpu_util)
        model = None
    else:
        tokenizer, model = load_model(model_path, load_in_4bit=args.load_in_4bit)
        llm = None
    print("Model loaded.")

    benchmarks = ALL_BENCHMARKS if args.benchmark == "all" else [args.benchmark]
    if args.data_path and args.benchmark == "all":
        raise ValueError("--data_path cannot be used with --benchmark all")

    summary_buf = io.StringIO()

    def log(msg=""):
        print(msg)
        summary_buf.write(msg + "\n")

    log(f"\n{'='*70}")
    log(f"  Benchmark Evaluation")
    log(f"  Model: {model_path}")
    log(f"  Mode: {original_mode} ({args.mode})" if original_mode != args.mode else f"  Mode: {args.mode}")
    log(f"  Tag: {tag}")
    log(f"  max_new_tokens: {max_new_tokens}")
    log(f"{'='*70}\n")

    all_stats = {}

    for bench in benchmarks:
        if args.data_path:
            path = Path(args.data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            data = load_jsonl(path)
            log(f"[{bench}] Loaded {len(data)} questions from {path}")
        else:
            data = load_jsonl(resolve_data_path(bench))
            log(f"[{bench}] Loaded {len(data)} questions")

        if args.mode != "parametric" and data and "retrieved_passages" not in data[0]:
            raise ValueError(
                "This evaluation mode requires raw retrieved document text in the "
                "`retrieved_passages` field. The public files only include retrieval IDs; "
                "reconstruct full contexts from the retrieval corpus or pass a full-context "
                "JSONL via --data_path."
            )

        if args.sample_n and args.sample_n < len(data):
            data = stratified_sample(data, args.sample_n, benchmark=bench, seed=seed)
            log(f"[{bench}] Subsampled to {len(data)} (stratified)")

        sample_tag = f"_n{len(data)}" if args.sample_n else ""
        out_path = output_dir / f"eval_{tag}_{bench}{sample_tag}.jsonl"

        if use_vllm:
            acc, stats = run_eval_vllm(
                data, tokenizer, llm, args.mode, out_path, max_new_tokens, bench,
                print_each=args.print_each, print_only_wrong=args.print_only_wrong)
        else:
            acc, stats = run_eval(
                data, tokenizer, model, args.mode, out_path, max_new_tokens,
                args.flush_every, bench,
                print_each=args.print_each, print_only_wrong=args.print_only_wrong)
        all_stats[bench] = stats

        log(f"\n[{bench}] Accuracy: {acc:.4f} ({stats['correct']}/{stats['total']})")
        log(f"[{bench}] Predictions: {stats['pred_counts']}")
        if args.mode in STRUCTURED_MODES:
            log(f"[{bench}] Decisions: {stats['decision_counts']}")
        if args.mode == "stage2":
            log(f"[{bench}] <ANALYSIS>: {stats['has_analysis']}/{stats['total']}")
            log(f"[{bench}] KG: {stats['has_kg']}/{stats['total']} "
                f"(empty: {stats['empty_kg']}, "
                f"avg E={stats['avg_entities']}, R={stats['avg_relations']})")
        log(f"[{bench}] Saved: {out_path}\n")

    # Summary
    log(f"\n{'='*70}")
    log(f"  SUMMARY — {tag}")
    log(f"{'='*70}")
    for bench, stats in all_stats.items():
        acc = stats["accuracy"]
        c, t = stats["correct"], stats["total"]
        o = stats["pred_counts"].get("other", 0)
        line = f"  {bench:<16} {c}/{t} ({acc:.2%})  [other={o}]"
        if args.mode in STRUCTURED_MODES:
            dc = stats['decision_counts']
            line += f"  [G={dc.get('grounded',0)} E={dc.get('elimination',0)} P={dc.get('parametric',0)}]"
        if args.mode == "stage2":
            line += f"  [kg={stats['has_kg']}/{t}]"
        log(line)
    log(f"{'='*70}\n")

    summary_path = output_dir / f"summary_{tag}_{args.benchmark}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_buf.getvalue())
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
