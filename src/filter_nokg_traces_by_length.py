#!/usr/bin/env python3
"""
filter_nokg_traces_by_length.py

Filter samples by token length for strict no-KG control training.
Token counts are computed with analysis_only prompt + analysis-only assistant target,
matching train_nokg_control.py.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from evaluate import ANALYSIS_ONLY_SYSTEM, ANALYSIS_ONLY_USER


# ── Same prompt as train_medjudgerag_mistral.py ──────────────
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

ANALYSIS_DELIMITER = "\n\n<ANALYSIS>\n"

# Strict no-KG control should match analysis_only prompt used at evaluation.
CONTROL_SYSTEM_PROMPT = ANALYSIS_ONLY_SYSTEM
CONTROL_USER_PROMPT_TEMPLATE = ANALYSIS_ONLY_USER


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


def normalize_analysis_only_target(answer_target):
    """Drop explicit KG lines from answer_target for strict no-KG control."""
    if not isinstance(answer_target, str):
        return ""
    out = []
    for line in answer_target.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("KG Entities:") or stripped.startswith("KG Relations:"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def build_full_training_text(item, tokenizer):
    """Build the exact text the model sees during training (prompt + assistant response)."""
    options = item.get("options", {})
    documents = build_documents(normalize_retrieved_docs(item))

    user_content = CONTROL_USER_PROMPT_TEMPLATE.format(
        question=item["question"],
        option_a=options.get("A", ""),
        option_b=options.get("B", ""),
        option_c=options.get("C", ""),
        option_d=options.get("D", ""),
        documents=documents,
    )

    # Strict no-KG control: assistant target is analysis-only completion text.
    answer_target = item.get("answer_target", "")
    assistant_content = normalize_analysis_only_target(answer_target)
    if not assistant_content:
        raise ValueError("empty assistant completion after no-KG normalization")

    messages = [
        {"role": "system", "content": CONTROL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Filter samples exceeding max sequence length (uses exact training prompt)"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model_name", required=True,
                        help="Model name or path for tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=8192,
                        help="Max sequence length (default: 8192)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Input: {len(samples)} samples")
    print(f"Max seq len: {args.max_seq_len}")

    kept = []
    dropped = []
    build_errors = []
    all_lens = []

    for item in samples:
        try:
            text = build_full_training_text(item, tokenizer)
            tok_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception as e:
            print(f"  ERROR building training text for {item.get('id', '?')}: {e}")
            build_errors.append((item.get("id", "?"), str(e)))
            continue

        all_lens.append(tok_len)
        if tok_len <= args.max_seq_len:
            kept.append(item)
        else:
            dropped.append((item["id"], tok_len))

    if not all_lens:
        raise RuntimeError(
            "No samples could be tokenized. If you used a public sanitized file, "
            "reconstruct a full-context trace file with `retrieved_docs` first."
        )

    train = sum(1 for i in kept if i.get("split") == "train")
    val = sum(1 for i in kept if i.get("split") == "val")

    with open(args.output, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lens = np.array(all_lens)
    print(f"\nToken length stats:")
    print(f"  mean={lens.mean():.0f}  median={np.median(lens):.0f}  "
          f"p95={np.percentile(lens, 95):.0f}  p99={np.percentile(lens, 99):.0f}  "
          f"max={lens.max()}")
    print(f"\nKept:         {len(kept)} (train={train}, val={val})")
    print(f"Dropped:      {len(dropped)} (>{args.max_seq_len} tokens)")
    print(f"Build errors: {len(build_errors)}")
    print(f"Output:       {args.output}")

    if dropped and len(dropped) <= 20:
        print(f"\nDropped samples (too long):")
        for sid, tl in dropped:
            print(f"  {sid}: {tl} tokens")


if __name__ == "__main__":
    main()
