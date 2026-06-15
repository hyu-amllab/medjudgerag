#!/usr/bin/env python3
"""
train_nokg_control.py

Strict no-KG control training script.
It trains only analysis completion (no KG prompt section, no KG assistant target section)
while preserving the same optimization stack as MedJudgeRAG training.

The default data/model arguments are Mistral-oriented. For Llama 3, pass
--model_name meta-llama/Meta-Llama-3-8B-Instruct and the corresponding
data/full_nokg_control_llama3_8192.jsonl files explicitly.
"""

import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from evaluate import ANALYSIS_ONLY_SYSTEM, ANALYSIS_ONLY_USER


# ── Constants ─────────────────────────────────────────────────────
ANALYSIS_DELIMITER = "\n\n<ANALYSIS>\n"
ANALYSIS_SEARCH_PATTERNS = ()

# ── Prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a biomedical information extractor and medical reasoner."

USER_PROMPT_TEMPLATE = """\
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

# Strict no-KG control prompt (must match analysis_only eval prompt).
CONTROL_SYSTEM_PROMPT = ANALYSIS_ONLY_SYSTEM
CONTROL_USER_PROMPT_TEMPLATE = ANALYSIS_ONLY_USER


# ── Data helpers ──────────────────────────────────────────────────
def build_documents(retrieved_docs):
    parts = []
    for i, doc in enumerate(retrieved_docs or []):
        content = doc.get("content", "")
        title = doc.get("title", "")
        if title:
            parts.append(f"[{i+1}] {title}. {content}")
        else:
            parts.append(f"[{i+1}] {content}")
    return "\n".join(parts)


def normalize_retrieved_docs(sample):
    """Return raw retrieved documents, or fail clearly for sanitized releases."""
    docs = sample.get("retrieved_docs")
    if docs:
        return docs

    passages = sample.get("retrieved_passages")
    if passages:
        normalized = []
        for passage in passages:
            if isinstance(passage, dict):
                normalized.append(passage)
            else:
                normalized.append({"content": str(passage)})
        return normalized

    ids = sample.get("retrieved_doc_ids") or sample.get("retrieved_ids")
    id_hint = f" Found retrieval IDs only: {ids[:3]}..." if ids else ""
    raise ValueError(
        "Raw retrieved document text is required in `retrieved_docs` or "
        "`retrieved_passages`. The public sanitized files contain retrieval IDs "
        "only; reconstruct full-context traces from the retrieval corpus before "
        "training or length filtering." + id_hint
    )


def reconstruct_analysis_text(answer_target):
    """Build strict no-KG analysis completion text from answer_target."""
    if isinstance(answer_target, str):
        out = []
        for line in answer_target.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("KG Entities:") or stripped.startswith("KG Relations:"):
                continue
            out.append(line)
        return "\n".join(out).strip()

    if not isinstance(answer_target, dict):
        return ""

    # Reconstruct dict target without KG references.
    oa = answer_target.get("option_analysis", {})
    lines = []
    for opt in ["A", "B", "C", "D"]:
        entry = oa.get(opt, {})
        verdict = entry.get("verdict", "INSUFFICIENT")
        evidence = entry.get("evidence", "No relevant evidence found.")
        doc_ids = entry.get("doc_ids", [])

        lines.append(f"[{opt}] Doc: {json.dumps(doc_ids)}")
        lines.append(f"Evidence: {evidence}")
        lines.append(f"Verdict: {verdict}")
        lines.append("")

    decision = answer_target.get("decision_mode", "grounded")
    summary = answer_target.get("decision_summary", "")
    answer = answer_target.get("answer_choice", "")
    lines.append(f"Decision: {decision}")
    lines.append(f"Summary: {summary}")
    lines.append(f"Answer_choice: {answer}")

    return "\n".join(lines)


def build_chat_messages(sample):
    """Build chat messages for strict no-KG control training."""
    options = sample.get("options", {})
    documents = build_documents(normalize_retrieved_docs(sample))

    user_content = CONTROL_USER_PROMPT_TEMPLATE.format(
        question=sample["question"],
        documents=documents,
        option_a=options.get("A", ""),
        option_b=options.get("B", ""),
        option_c=options.get("C", ""),
        option_d=options.get("D", ""),
    )

    # Analysis text only (no KG prefix) for strict control.
    analysis_text = reconstruct_analysis_text(sample.get("answer_target", ""))
    assistant_content = analysis_text

    return [
        {"role": "system", "content": CONTROL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def render_training_text(sample, tokenizer):
    messages = build_chat_messages(sample)
    return tokenizer.apply_chat_template(messages, tokenize=False)


def _validate_answer_target(answer_target):
    """Check that answer_target has meaningful content for training."""
    if isinstance(answer_target, str):
        text = answer_target.strip()
        if not text:
            return False
        for opt in ["[A]", "[B]", "[C]", "[D]"]:
            if opt not in text:
                return False
        if "Answer_choice:" not in text:
            return False
        return True

    if isinstance(answer_target, dict):
        oa = answer_target.get("option_analysis", {})
        if not all(opt in oa for opt in ["A", "B", "C", "D"]):
            return False
        if not answer_target.get("answer_choice"):
            return False
        return True

    return False


def load_data(data_path, split_filter=None):
    """Load JSONL data for no-KG control SFT.

    Args:
        data_path: Path to JSONL file.
        split_filter: If set (e.g. "train" or "val"), only keep samples whose
                      ``split`` field matches.
    """
    samples = []
    skipped = 0
    filtered = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if split_filter and item.get("split") != split_filter:
                filtered += 1
                continue
            if "answer_target" not in item:
                skipped += 1
                continue
            if not _validate_answer_target(item["answer_target"]):
                skipped += 1
                continue
            try:
                item["retrieved_docs"] = normalize_retrieved_docs(item)
            except ValueError as exc:
                raise ValueError(
                    f"{data_path}: sample {item.get('id', '?')} cannot be used for "
                    f"document-conditioned training. {exc}"
                ) from exc
            samples.append(item)
    if skipped > 0:
        print(f"  WARNING: {skipped} samples dropped (missing/incomplete answer_target)")
    if filtered > 0:
        print(f"  Filtered {filtered} samples (split != {split_filter!r})")
    return samples


def formatting_func(examples, tokenizer):
    """Format examples into chat template strings for SFTTrainer."""
    texts = []
    for i in range(len(examples["question"])):
        sample = {
            "question": examples["question"][i],
            "options": examples["options"][i],
            "retrieved_docs": examples["retrieved_docs"][i],
            "kg": examples["kg"][i],
            "answer_target": examples["answer_target"][i],
        }
        text = render_training_text(sample, tokenizer)
        texts.append(text)
    return texts


# ── Utility functions ─────────────────────────────────────────────
def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * p
    lower = int(math.floor(k))
    upper = int(math.ceil(k))
    if lower == upper:
        return float(sorted_values[lower])
    frac = k - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _find_subsequence(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return -1
    last = len(sequence) - len(pattern) + 1
    for i in range(last):
        if sequence[i : i + len(pattern)] == pattern:
            return i
    return -1


def _find_subsequence_from(sequence, pattern, start=0):
    if not pattern or len(pattern) > len(sequence):
        return -1
    start = max(0, start)
    last = len(sequence) - len(pattern) + 1
    for i in range(start, last):
        if sequence[i : i + len(pattern)] == pattern:
            return i
    return -1


def find_analysis_delimiter(ids, assistant_header_ids, analysis_pattern_ids_list):
    """Find the <ANALYSIS> delimiter inside the assistant span.

    Returns (analysis_pos, matched_pattern_ids, assistant_pos).
    """
    assistant_pos = -1
    search_start = 0
    if assistant_header_ids:
        assistant_pos = _find_subsequence(ids, assistant_header_ids)
        if assistant_pos != -1:
            search_start = assistant_pos + len(assistant_header_ids)

    best_pos = -1
    best_pattern_ids = None
    for pattern_ids in analysis_pattern_ids_list:
        pos = _find_subsequence_from(ids, pattern_ids, start=search_start)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_pattern_ids = pattern_ids

    return best_pos, best_pattern_ids, assistant_pos


def find_boundaries(ids, assistant_header_ids, analysis_pattern_ids_list):
    """Find prompt_end and analysis_start token positions.

    Returns:
        prompt_end: first token of assistant content (after assistant header).
                    Everything before this is prompt (loss=0).
        analysis_start: first token of <ANALYSIS> delimiter.
                    prompt_end..analysis_start = KG region (loss=kg_weight).
                    analysis_start..end = analysis region (loss=analysis_weight).
        Returns (-1, -1) if boundaries cannot be found.
    """
    if assistant_header_ids:
        assistant_pos = _find_subsequence(ids, assistant_header_ids)
    else:
        assistant_pos = -1

    if assistant_pos == -1:
        return -1, -1

    prompt_end = assistant_pos + len(assistant_header_ids)
    # Strict no-KG control: no dedicated KG segment, whole assistant span is analysis region.
    analysis_start = prompt_end
    return prompt_end, analysis_start


def _longest_common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def infer_response_template_ids(tokenizer):
    """Infer assistant header token IDs from tokenizer chat template."""
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "User."},
    ]

    no_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    with_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    marker = ""
    if with_gen.startswith(no_gen):
        marker = with_gen[len(no_gen):]
    else:
        lcp = _longest_common_prefix_len(no_gen, with_gen)
        marker = with_gen[lcp:]

    if not marker.strip():
        dummy = "__ASSISTANT_DUMMY_OUTPUT__"
        with_dummy = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": dummy}],
            tokenize=False,
            add_generation_prompt=False,
        )
        pos = with_dummy.find(dummy)
        if pos != -1:
            prefix = with_dummy[:pos]
            if prefix.startswith(no_gen):
                candidate = prefix[len(no_gen):]
            else:
                lcp = _longest_common_prefix_len(no_gen, prefix)
                candidate = prefix[lcp:]
            if candidate.strip():
                marker = candidate

    if not marker.strip():
        if "[/INST]" in no_gen:
            marker = "[/INST]"
        else:
            raise RuntimeError(
                "Failed to infer response template from chat template. "
                "Consider adding a --response_template CLI override."
            )

    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if not marker_ids:
        raise RuntimeError("Failed to infer response template IDs.")

    return marker, marker_ids


# ── Weighted Loss Collator ────────────────────────────────────────
class WeightedLossCollator:
    """Data collator that produces per-token loss_weights.

    Loss regions:
        [system + user prompt]  → labels=-100 (no loss)
        [assistant completion]  → loss_weights=analysis_weight
        [padding]               → labels=-100 (no loss)

    Samples where boundaries cannot be detected are skipped (fully masked).
    """

    def __init__(self, tokenizer, assistant_header_ids, analysis_pattern_ids_list,
                 kg_weight=0.0, analysis_weight=1.0, ignore_index=-100):
        self.tokenizer = tokenizer
        self.assistant_header_ids = list(assistant_header_ids)
        self.analysis_pattern_ids_list = [list(x) for x in analysis_pattern_ids_list]
        self.kg_weight = kg_weight
        self.analysis_weight = analysis_weight
        self.ignore_index = ignore_index

    def __call__(self, features):
        cleaned = []
        allowed_keys = {"input_ids", "attention_mask", "labels"}
        for feature in features:
            cleaned.append({k: v for k, v in feature.items() if k in allowed_keys})

        batch = self.tokenizer.pad(cleaned, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        bsz, seq_len = input_ids.shape
        loss_weights = torch.zeros(bsz, seq_len, dtype=torch.float32)

        n_ok = 0
        n_skipped = 0

        for i in range(bsz):
            ids = input_ids[i].tolist()

            prompt_end, analysis_start = find_boundaries(
                ids,
                self.assistant_header_ids,
                self.analysis_pattern_ids_list,
            )

            if prompt_end == -1 or analysis_start == -1:
                # Cannot find boundaries → fully mask
                labels[i, :] = self.ignore_index
                n_skipped += 1
                continue

            # Prompt region: mask labels
            labels[i, :prompt_end] = self.ignore_index

            # KG region: kg_weight
            loss_weights[i, prompt_end:analysis_start] = self.kg_weight

            # Analysis region: analysis_weight
            loss_weights[i, analysis_start:] = self.analysis_weight

            n_ok += 1

            # Mask padding tokens
            if self.tokenizer.pad_token_id is not None:
                pad_mask = (input_ids[i] == self.tokenizer.pad_token_id)
                labels[i, pad_mask] = self.ignore_index
                loss_weights[i, pad_mask] = 0.0

        batch["labels"] = labels
        batch["loss_weights"] = loss_weights

        if not getattr(self, "_warned", False):
            if n_skipped > 0:
                print(
                    f"[WeightedLossCollator] WARNING: {n_skipped}/{bsz} "
                    f"samples fully masked (boundaries not found)"
                )
            self._warned = True

        return batch


# ── Precision helpers ─────────────────────────────────────────────
def resolve_precision():
    cuda_available = torch.cuda.is_available()
    bf16_supported = False
    if cuda_available and hasattr(torch.cuda, "is_bf16_supported"):
        bf16_supported = bool(torch.cuda.is_bf16_supported())

    if cuda_available and bf16_supported:
        return torch.bfloat16, torch.bfloat16, True, False, "bf16"
    if cuda_available:
        return torch.float16, torch.float16, False, True, "fp16"
    return torch.float32, torch.float32, False, False, "fp32"


def _flag_enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def resolve_precision_with_deepspeed(args, default_tuple):
    model_dtype, compute_dtype, use_bf16, use_fp16, precision_name = default_tuple

    if not args.deepspeed:
        return default_tuple

    ds_path = Path(args.deepspeed)
    if not ds_path.exists():
        print(f"DeepSpeed config not found at {ds_path}; keeping auto precision: {precision_name}")
        return default_tuple

    try:
        ds_cfg = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to parse DeepSpeed config ({ds_path}): {exc}. Keeping auto precision.")
        return default_tuple

    ds_fp16 = _flag_enabled(ds_cfg.get("fp16", {}).get("enabled", False))
    ds_bf16 = _flag_enabled(ds_cfg.get("bf16", {}).get("enabled", False))

    if ds_fp16 and ds_bf16:
        raise ValueError(f"Invalid DeepSpeed config: both fp16 and bf16 enabled.")

    if ds_fp16:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
        return torch.float16, torch.float16, False, True, "fp16 (deepspeed)"
    if ds_bf16:
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
        return torch.bfloat16, torch.bfloat16, True, False, "bf16 (deepspeed)"

    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    return torch.float32, torch.float32, False, False, "fp32 (deepspeed)"


def _verify_adapter_weights(output_dir):
    adapter_path = Path(output_dir) / "adapter_model.safetensors"
    if not adapter_path.exists():
        print(f"  [verify] adapter file not found at {adapter_path}, skipping check.")
        return
    try:
        from safetensors import safe_open
        empty_keys = []
        with safe_open(str(adapter_path), framework="pt") as f:
            for key in f.keys():
                if f.get_tensor(key).numel() == 0:
                    empty_keys.append(key)
        if empty_keys:
            print(f"  [verify] WARNING: {len(empty_keys)} empty tensor(s) in adapter!")
            for k in empty_keys[:5]:
                print(f"    - {k}")
        else:
            print(f"  [verify] adapter OK – no empty tensors found.")
    except ImportError:
        print("  [verify] safetensors not installed, skipping adapter verification.")
    except Exception as exc:
        print(f"  [verify] adapter check failed: {exc}")


# ── Debug & Stats ─────────────────────────────────────────────────
def log_sample_debug(samples, tokenizer, assistant_header_ids,
                     analysis_pattern_ids_list, n=3):
    """Print boundary debug info for the first n samples."""
    print(f"\n{'='*60}")
    print(f"[DEBUG] Boundary detection on first {n} samples:")
    print(f"{'='*60}")
    for idx, sample in enumerate(samples[:n]):
        text = render_training_text(sample, tokenizer)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_end, analysis_start = find_boundaries(
            ids, assistant_header_ids, analysis_pattern_ids_list
        )
        total = len(ids)
        sid = sample.get("id", "<no-id>")

        if prompt_end == -1:
            print(f"  [{idx}] id={sid}  total={total}  "
                  f"BOUNDARIES NOT FOUND")
            continue

        kg_tokens = analysis_start - prompt_end
        analysis_tokens = total - analysis_start
        print(f"  [{idx}] id={sid}  total={total}  "
              f"prompt={prompt_end}  kg={kg_tokens}  analysis={analysis_tokens}  "
              f"found_assistant=yes  found_analysis=yes")
    print()


def log_dataset_stats(samples, tokenizer, assistant_header_ids,
                      analysis_pattern_ids_list, max_seq_len, split_name):
    """Print dataset-level token statistics."""
    if not samples:
        print(f"[{split_name}] no samples")
        return

    total_lengths = []
    kg_lengths = []
    analysis_lengths = []
    n_skipped = 0
    truncated = 0

    for sample in samples:
        text = render_training_text(sample, tokenizer)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        total_len = len(ids)
        total_lengths.append(total_len)

        if total_len > max_seq_len:
            truncated += 1

        prompt_end, analysis_start = find_boundaries(
            ids, assistant_header_ids, analysis_pattern_ids_list
        )
        if prompt_end == -1:
            n_skipped += 1
            continue

        kg_lengths.append(analysis_start - prompt_end)
        analysis_lengths.append(total_len - analysis_start)

    sorted_total = sorted(total_lengths)
    n = len(sorted_total)
    trunc_pct = 100.0 * truncated / n if n else 0

    print(f"\n[{split_name}] Dataset statistics:")
    print(f"  samples: {n}  (skipped/no-boundary: {n_skipped})")
    print(f"  truncated (>{max_seq_len}): {truncated} ({trunc_pct:.1f}%)")
    print(f"  total tokens: min={sorted_total[0]} p50={percentile(sorted_total, 0.5):.0f} "
          f"p95={percentile(sorted_total, 0.95):.0f} max={sorted_total[-1]}")

    if kg_lengths:
        sorted_kg = sorted(kg_lengths)
        sorted_an = sorted(analysis_lengths)
        print(f"  KG tokens:    mean={sum(kg_lengths)/len(kg_lengths):.0f} "
              f"p50={percentile(sorted_kg, 0.5):.0f} "
              f"p95={percentile(sorted_kg, 0.95):.0f}")
        print(f"  Analysis tok: mean={sum(analysis_lengths)/len(analysis_lengths):.0f} "
              f"p50={percentile(sorted_an, 0.5):.0f} "
              f"p95={percentile(sorted_an, 0.95):.0f}")
    print()


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Strict no-KG control SFT (analysis-only completion)"
    )
    parser.add_argument("--train_data", default="data/full_nokg_control_mistral_8192.jsonl")
    parser.add_argument("--val_data", default="data/full_nokg_control_mistral_8192.jsonl")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Base model (default: Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--output_dir", default="outputs/nokg_control_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--kg_weight", type=float, default=0.0,
                        help="No-KG mode: kept for logging compatibility (default: 0.0)")
    parser.add_argument("--analysis_weight", type=float, default=1.0,
                        help="Loss weight for analysis region (default: 1.0)")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--save_strategy", choices=["epoch", "no"], default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--response_template", type=str, default=None)
    args = parser.parse_args()

    if args.no_4bit:
        args.use_4bit = False

    is_distributed = any(k in os.environ for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    print(f"Loss weights: KG={args.kg_weight}, Analysis={args.analysis_weight}")
    if args.kg_weight != 0.0:
        print("WARNING: strict no-KG mode has no KG token span, so kg_weight has no effect.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    # Load data
    print("Loading training data...")
    same_file = (args.train_data == args.val_data)
    train_samples = load_data(args.train_data, split_filter="train" if same_file else None)
    print(f"  Train: {len(train_samples)} samples")

    val_samples = None
    if args.val_data and Path(args.val_data).exists():
        val_samples = load_data(args.val_data, split_filter="val" if same_file else None)
        print(f"  Val: {len(val_samples)} samples")

    # Infer assistant header
    if args.response_template:
        response_template_text = args.response_template
        assistant_header_ids = tokenizer.encode(response_template_text, add_special_tokens=False)
        print(f"Response template (manual override): {response_template_text!r}")
    else:
        response_template_text, assistant_header_ids = infer_response_template_ids(tokenizer)
        print(f"Response template (auto-inferred): {response_template_text!r}")

    # Strict no-KG mode: boundaries are assistant-start only (no <ANALYSIS> split).
    analysis_pattern_ids_list = []
    print("Strict no-KG mode: using assistant-start boundary only.")

    # Debug: boundary detection on first samples
    log_sample_debug(train_samples, tokenizer, assistant_header_ids,
                     analysis_pattern_ids_list, n=3)

    # Hard guard: do not run training when prompt/completion boundaries are missing.
    check_n = min(64, len(train_samples))
    missing = 0
    for sample in train_samples[:check_n]:
        ids = tokenizer(
            render_training_text(sample, tokenizer),
            add_special_tokens=False
        )["input_ids"]
        prompt_end, analysis_start = find_boundaries(
            ids, assistant_header_ids, analysis_pattern_ids_list
        )
        if prompt_end == -1 or analysis_start == -1:
            missing += 1
    if missing > 0:
        raise RuntimeError(
            f"Boundary detection failed for {missing}/{check_n} probe samples. "
            "Please pass a correct --response_template (e.g., '[/INST]' for Mistral)."
        )

    # Dataset statistics
    log_dataset_stats(train_samples, tokenizer, assistant_header_ids,
                      analysis_pattern_ids_list, args.max_seq_len, "train")
    if val_samples:
        log_dataset_stats(val_samples, tokenizer, assistant_header_ids,
                          analysis_pattern_ids_list, args.max_seq_len, "val")

    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples) if val_samples else None

    # Precision
    precision_tuple = resolve_precision()
    model_dtype, compute_dtype, use_bf16, use_fp16, precision_name = resolve_precision_with_deepspeed(
        args, precision_tuple
    )
    print(f"Precision mode: {precision_name} (model_dtype={model_dtype}, compute_dtype={compute_dtype})")

    if torch.cuda.is_available() and not is_distributed:
        torch.cuda.set_device(0)
        print(
            f"CUDA visible count: {torch.cuda.device_count()}, "
            f"using device 0 -> {torch.cuda.get_device_name(0)}"
        )
    if torch.cuda.is_available() and is_distributed:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        visible = torch.cuda.device_count()
        if world_size > visible:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} exceeds visible CUDA devices={visible}."
            )

    if args.use_4bit and not torch.cuda.is_available():
        print("CUDA not available; disabling 4-bit quantization.")
        args.use_4bit = False

    # Model
    print(f"Loading model: {args.model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            model_kwargs["device_map"] = {"": local_rank}
            print(f"4-bit distributed load: device_map={{'': {local_rank}}}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Collator
    collator = WeightedLossCollator(
        tokenizer=tokenizer,
        assistant_header_ids=assistant_header_ids,
        analysis_pattern_ids_list=analysis_pattern_ids_list,
        kg_weight=args.kg_weight,
        analysis_weight=args.analysis_weight,
    )
    print(f"Using WeightedLossCollator (kg_weight={args.kg_weight}, "
          f"analysis_weight={args.analysis_weight})")

    # Training args
    load_best = bool(val_dataset and args.save_strategy != "no")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        per_device_eval_batch_size=args.batch,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    # Trainer: use WeightedLossTrainer with SFTTrainer's data pipeline
    # SFTTrainer handles formatting_func → tokenization; we override its
    # Trainer base with our WeightedLossTrainer for the loss computation.

    # We need a combined class that has SFTTrainer's data handling
    # but WeightedLossTrainer's compute_loss.
    class WeightedSFTTrainer(SFTTrainer):
        """SFTTrainer with per-token weighted loss."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss_weights = inputs.pop("loss_weights", None)

            outputs = model(**inputs)

            if loss_weights is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            logits = outputs.logits
            labels = inputs["labels"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_weights = loss_weights[:, 1:].contiguous().to(shift_logits.device)

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            flat_weights = shift_weights.view(-1)

            ce = F.cross_entropy(flat_logits, flat_labels, reduction="none",
                                 ignore_index=-100)

            valid_mask = (flat_labels != -100)
            weighted_ce = ce * flat_weights * valid_mask.float()

            weight_sum = (flat_weights * valid_mask.float()).sum()
            if weight_sum > 0:
                loss = weighted_ce.sum() / weight_sum
            else:
                loss = weighted_ce.sum()

            return (loss, outputs) if return_outputs else loss

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        formatting_func=lambda examples: formatting_func(examples, tokenizer=tokenizer),
        max_seq_length=args.max_seq_len,
    )
    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedSFTTrainer(**trainer_kwargs)

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)

    if trainer.is_world_process_zero():
        _verify_adapter_weights(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
