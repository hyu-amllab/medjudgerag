#!/usr/bin/env python3
"""
generate_teacher_traces.py — Generate MedJudgeRAG teacher traces via GPT-5.1 Batch API.

GPT receives (Q, D, O) and generates KG + <ANALYSIS> + verdicts + Decision + Answer_choice
in a single pass. No KG is provided as input — GPT builds its own KG.

Gold answer is NOT provided to GPT. We validate afterwards (answer_choice == gold).

Input:  data/source_train.jsonl / data/source_val.jsonl
        (source files must contain Q, options, answer, split, and retrieved documents.)
Output: teacher_trace_batches/{batch_id}/results.jsonl

Workflow:
    python generate_teacher_traces.py prepare --batch_id 0 --offset 0 --limit 250
    python generate_teacher_traces.py submit  --batch_id 0
    python generate_teacher_traces.py status  --batch_id 0
    python generate_teacher_traces.py download --batch_id 0
    python generate_teacher_traces.py merge
    python generate_teacher_traces.py validate
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_INPUT = PROJECT_DIR / "data" / "source_train.jsonl"
VAL_INPUT = PROJECT_DIR / "data" / "source_val.jsonl"
BATCH_BASE_DIR = PROJECT_DIR / "teacher_trace_batches"
OUTPUT_DIR = PROJECT_DIR / "data"


# ── Prompt (MedJudgeRAG end-to-end: compact KG rules + analysis rules) ─
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


# ── Helpers ────────────────────────────────────────────────────────
def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key


def load_data(train_input, val_input):
    samples = []
    for fpath, split in [(train_input, "train"), (val_input, "val")]:
        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping.")
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["_split"] = split
                    samples.append(item)
    return samples


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
        "only; reconstruct full-context inputs from the retrieval corpus before "
        "teacher trace generation." + id_hint
    )


def get_batch_dir(batch_id):
    return BATCH_BASE_DIR / f"batch_{batch_id}"


# ── Validation ─────────────────────────────────────────────────────
VALID_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}
VALID_DECISIONS = {"grounded", "elimination", "parametric"}


def parse_full_output(text):
    """Parse GPT's end-to-end output (KG + <ANALYSIS> + verdicts + decision + answer).

    Returns dict with keys: kg, option_analysis, decision_mode, decision_summary, answer_choice
    """
    if not isinstance(text, str) or not text.strip():
        return None

    result = {}

    # Split at <ANALYSIS> — required delimiter
    analysis_pos = text.find("<ANALYSIS>")
    if analysis_pos < 0:
        return None  # <ANALYSIS> missing → invalid output
    result["kg"] = text[:analysis_pos].strip()
    analysis_text = text[analysis_pos + len("<ANALYSIS>"):].strip()

    # Parse option blocks from analysis
    result["option_analysis"] = {}
    for opt in ["A", "B", "C", "D"]:
        opt_m = re.search(rf'\[{opt}\]\s*', analysis_text)
        if not opt_m:
            continue
        block_start = opt_m.end()
        next_m = re.search(rf'(?:\[[ABCD]\]|Decision:)', analysis_text[block_start:])
        block_end = block_start + next_m.start() if next_m else len(analysis_text)
        block = analysis_text[block_start:block_end]

        doc_m = re.search(r'Doc:\s*(\[.*?\])', block)
        try:
            doc_ids = json.loads(doc_m.group(1)) if doc_m else []
            if not isinstance(doc_ids, list):
                doc_ids = []
        except json.JSONDecodeError:
            doc_ids = []

        # New format (preferred): KG Entities / KG Relations
        kg_entities = []
        kg_relations = []

        kge_m = re.search(r'KG Entities:\s*(\[.*?\])', block, re.DOTALL)
        if kge_m:
            try:
                parsed = json.loads(kge_m.group(1))
                if isinstance(parsed, list):
                    kg_entities = [x for x in parsed if isinstance(x, str)]
            except json.JSONDecodeError:
                kg_entities = []

        kgr_m = re.search(r'KG Relations:\s*(\[.*?\])', block, re.DOTALL)
        if kgr_m:
            kgr_raw = kgr_m.group(1).strip()
            if kgr_raw != "[]":
                # Accept R1 / R2 / Ri style IDs.
                kg_relations = re.findall(r'\bR[A-Za-z0-9_]+\b', kgr_raw)

        # Legacy format fallback: KG: [("Entity"...), ("Relation"...)]
        kg_m = re.search(r'KG:\s*(\[.*?)(?:\nEvidence:|\nVerdict:|\Z)', block, re.DOTALL)
        if kg_m:
            kg_raw = kg_m.group(1).strip()
            kg_refs = [] if kg_raw == "[]" else re.findall(r'(\("(?:Entity|Relation)"[^)]*\))', kg_raw)
        else:
            kg_refs = []

        ev_m = re.search(r'Evidence:\s*(.+?)(?:\nVerdict:|\Z)', block, re.DOTALL)
        evidence = ev_m.group(1).strip() if ev_m else ""

        v_m = re.search(r'Verdict:\s*(SUPPORTED|CONTRADICTED|INSUFFICIENT)', block)
        verdict = v_m.group(1) if v_m else ""

        result["option_analysis"][opt] = {
            "verdict": verdict,
            "evidence": evidence,
            "doc_ids": doc_ids,
            "kg_entities": kg_entities,
            "kg_relations": kg_relations,
            "kg_refs": kg_refs,
        }

    dec_m = re.search(r'Decision:\s*(grounded|elimination|parametric)', analysis_text)
    result["decision_mode"] = dec_m.group(1) if dec_m else ""

    sum_m = re.search(r'Summary:\s*(.+?)(?:\nAnswer_choice:|\nAnswer:|\Z)', analysis_text, re.DOTALL)
    result["decision_summary"] = sum_m.group(1).strip() if sum_m else ""

    ans_m = re.search(r'Answer_choice:\s*([A-D])', analysis_text) or \
            re.search(r'Answer:\s*([A-D])', analysis_text)
    result["answer_choice"] = ans_m.group(1) if ans_m else ""

    return result


def validate_target(parsed, gold):
    """Validate parsed target. Returns (pass, reason)."""
    if not parsed:
        return False, "parse_fail"

    if parsed.get("answer_choice") != gold:
        return False, f"wrong_answer: chose {parsed.get('answer_choice')}, gold={gold}"

    oa = parsed.get("option_analysis", {})
    for opt in ["A", "B", "C", "D"]:
        if opt not in oa:
            return False, f"missing_option_{opt}"

    decision = parsed.get("decision_mode", "")
    if decision not in VALID_DECISIONS:
        return False, f"invalid_decision: {decision}"

    # Semantic checks
    gold_verdict = oa.get(gold, {}).get("verdict", "")
    if decision == "grounded" and gold_verdict != "SUPPORTED":
        return False, f"grounded_gold_not_supported: {gold_verdict}"

    non_gold_supported = [o for o in ["A", "B", "C", "D"]
                          if o != gold and oa.get(o, {}).get("verdict") == "SUPPORTED"]
    if non_gold_supported:
        return False, f"non_gold_supported: {non_gold_supported}"

    if decision == "elimination":
        if gold_verdict == "CONTRADICTED":
            return False, "elimination_gold_contradicted"
        has_supported = any(oa[o].get("verdict") == "SUPPORTED" for o in ["A", "B", "C", "D"])
        has_contradicted = any(oa[o].get("verdict") == "CONTRADICTED" for o in ["A", "B", "C", "D"])
        if has_supported:
            return False, "elimination_but_supported_exists"
        if not has_contradicted:
            return False, "elimination_but_no_contradicted"

    if decision == "parametric":
        for o in ["A", "B", "C", "D"]:
            v = oa.get(o, {}).get("verdict", "")
            if v != "INSUFFICIENT":
                return False, f"parametric_not_all_insufficient: {o}={v}"

    return True, "pass"


# ── Subcommands ────────────────────────────────────────────────────
def cmd_prepare(args):
    """Build batch input for GPT end-to-end generation."""
    batch_dir = get_batch_dir(args.batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)

    samples = load_data(Path(args.train_input), Path(args.val_input))
    total = len(samples)

    if args.offset > 0:
        samples = samples[args.offset:]
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Total: {total}, Processing: {len(samples)} (offset={args.offset}, limit={args.limit})")

    input_path = batch_dir / "input.jsonl"
    meta_path = batch_dir / "meta.jsonl"
    count = 0

    with input_path.open("w", encoding="utf-8") as f_in, \
         meta_path.open("w", encoding="utf-8") as f_meta:

        for item in samples:
            options = item.get("options", {})
            retrieved_docs = normalize_retrieved_docs(item)
            documents = build_documents(retrieved_docs)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=item["question"],
                option_a=options.get("A", ""),
                option_b=options.get("B", ""),
                option_c=options.get("C", ""),
                option_d=options.get("D", ""),
                documents=documents,
            )

            custom_id = f"e2e-{item['id']}"

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                },
            }
            f_in.write(json.dumps(request, ensure_ascii=False) + "\n")

            meta = {
                "custom_id": custom_id,
                "id": item["id"],
                "benchmark": item.get("benchmark", ""),
                "question": item["question"],
                "options": options,
                "answer": item["answer"],
                "split": item["_split"],
                "retrieved_docs": retrieved_docs,
            }
            f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1

    file_size_mb = input_path.stat().st_size / 1024 / 1024
    print(f"[prepare] {count} requests → {input_path} ({file_size_mb:.1f} MB)")


def cmd_submit(args):
    """Upload and create batch job."""
    batch_dir = get_batch_dir(args.batch_id)
    input_path = batch_dir / "input.jsonl"
    info_path = batch_dir / "batch_info.json"

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run 'prepare' first.")

    client = OpenAI(api_key=load_api_key())

    print("[submit] Uploading...")
    with input_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"[submit] File: {file_obj.id}")

    n_requests = sum(1 for _ in input_path.open())
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"E2E SFT targets batch_{args.batch_id} ({n_requests} samples)"},
    )
    print(f"[submit] Batch: {batch.id} (status: {batch.status})")

    info = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "status": batch.status,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def cmd_status(args):
    """Check batch status."""
    batch_dir = get_batch_dir(args.batch_id)
    info_path = batch_dir / "batch_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    client = OpenAI(api_key=load_api_key())

    batch = client.batches.retrieve(info["batch_id"])
    print(f"[status] batch_{args.batch_id}: {batch.status}")
    print(f"  Total: {batch.request_counts.total}")
    print(f"  Completed: {batch.request_counts.completed}")
    print(f"  Failed: {batch.request_counts.failed}")

    info["status"] = batch.status
    if batch.output_file_id:
        info["output_file_id"] = batch.output_file_id
    if batch.error_file_id:
        info["error_file_id"] = batch.error_file_id
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return batch.status


def cmd_download(args):
    """Download results for a single batch."""
    batch_dir = get_batch_dir(args.batch_id)
    info_path = batch_dir / "batch_info.json"
    meta_path = batch_dir / "meta.jsonl"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    client = OpenAI(api_key=load_api_key())

    batch = client.batches.retrieve(info["batch_id"])
    if batch.status != "completed":
        print(f"[download] Not completed: {batch.status}")
        return

    raw_path = batch_dir / "raw_output.jsonl"
    content = client.files.content(batch.output_file_id)
    raw_path.write_bytes(content.read())
    print(f"[download] Raw output → {raw_path}")

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        (batch_dir / "errors.jsonl").write_bytes(error_content.read())

    # Load metadata
    meta_map = {}
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            meta_map[m["custom_id"]] = m

    # Parse results
    results_path = batch_dir / "results.jsonl"
    counts = {"total": 0, "correct": 0, "error": 0}

    with raw_path.open(encoding="utf-8") as f_raw, \
         results_path.open("w", encoding="utf-8") as f_out:

        for line in f_raw:
            result = json.loads(line)
            custom_id = result["custom_id"]
            meta = meta_map.get(custom_id)

            if meta is None:
                counts["error"] += 1
                continue

            response = result.get("response", {})
            if response.get("status_code") != 200:
                print(f"  ERROR {custom_id}: status {response.get('status_code')}")
                counts["error"] += 1
                continue

            body = response.get("body", {})
            choices = body.get("choices", [])
            full_output = choices[0]["message"]["content"].strip() if choices else ""

            # Split output into KG and answer_target at <ANALYSIS>
            analysis_pos = full_output.find("<ANALYSIS>")
            if analysis_pos >= 0:
                kg_text = full_output[:analysis_pos].strip()
                answer_target = full_output[analysis_pos + len("<ANALYSIS>"):].strip()
            else:
                kg_text = ""
                answer_target = full_output

            # Check correctness
            ans_m = re.search(r'Answer_choice:\s*([A-D])', answer_target) or \
                    re.search(r'Answer:\s*([A-D])', answer_target)
            pred = ans_m.group(1) if ans_m else ""
            is_correct = pred == meta["answer"]

            out_obj = {
                "id": meta["id"],
                "benchmark": meta.get("benchmark", ""),
                "question": meta["question"],
                "options": meta["options"],
                "answer": meta["answer"],
                "split": meta["split"],
                "retrieved_docs": meta["retrieved_docs"],
                "kg": kg_text,
                "answer_target": answer_target,
                "full_output": full_output,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            counts["total"] += 1
            if is_correct:
                counts["correct"] += 1

    acc = counts["correct"] / max(counts["total"], 1) * 100
    print(f"[download] batch_{args.batch_id}: {counts['correct']}/{counts['total']} ({acc:.1f}%), errors={counts['error']}")
    print(f"  → {results_path}")


def cmd_merge(args):
    """Merge all batch results into a single file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "teacher_traces_merged.jsonl"

    batch_dirs = sorted(BATCH_BASE_DIR.glob("batch_*"))
    seen = set()
    all_results = []

    for bd in batch_dirs:
        rp = bd / "results.jsonl"
        if rp.exists():
            count = 0
            with rp.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if item["id"] not in seen:
                            seen.add(item["id"])
                            all_results.append(item)
                            count += 1
            print(f"  Loaded {bd.name}: {count} new")

    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    correct = sum(1 for r in all_results
                  if (re.search(r'Answer_choice:\s*([A-D])', r.get("answer_target", "")) or
                      re.search(r'Answer:\s*([A-D])', r.get("answer_target", "")))
                  and (re.search(r'Answer_choice:\s*([A-D])', r.get("answer_target", "")) or
                       re.search(r'Answer:\s*([A-D])', r.get("answer_target", ""))).group(1) == r["answer"])
    total = len(all_results)
    train = sum(1 for r in all_results if r.get("split") == "train")
    val = sum(1 for r in all_results if r.get("split") == "val")

    print(f"[merge] Total: {total} (train={train}, val={val})")
    print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  → {out_path}")


def cmd_validate(args):
    """Validate merged results and produce final clean dataset."""
    merged_path = OUTPUT_DIR / "teacher_traces_merged.jsonl"
    clean_path = OUTPUT_DIR / "teacher_traces_raw.jsonl"

    if not merged_path.exists():
        raise FileNotFoundError(f"{merged_path} not found. Run 'merge' first.")

    total = 0
    passed = 0
    fail_reasons = {}

    with merged_path.open(encoding="utf-8") as f_in, \
         clean_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1

            full_output = item.get("full_output", "")
            parsed = parse_full_output(full_output)
            gold = item.get("answer", "")

            ok, reason = validate_target(parsed, gold)
            if ok:
                passed += 1
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    train = 0
    val = 0
    with clean_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == "train":
                    train += 1
                else:
                    val += 1

    print(f"[validate] Total: {total}")
    print(f"  PASS: {passed} (train={train}, val={val})")
    print(f"  FAIL: {total - passed}")
    if fail_reasons:
        print(f"\n  Fail reasons:")
        for reason, cnt in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")
    print(f"\n  → {clean_path}")


def cmd_prepare_retry(args):
    """Build batch input for failed samples only (quota errors etc.)."""
    failed_ids_path = BATCH_BASE_DIR / "failed_ids.json"
    if not failed_ids_path.exists():
        raise FileNotFoundError(
            f"{failed_ids_path} not found. "
            "Run the failed ID collection script first."
        )

    failed_ids = set(json.loads(failed_ids_path.read_text()))
    print(f"[prepare_retry] Failed IDs to retry: {len(failed_ids)}")

    # Collect meta from original batches for failed IDs
    failed_meta = {}
    for bd in sorted(BATCH_BASE_DIR.glob("batch_*")):
        meta_path = bd / "meta.jsonl"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    if m["id"] in failed_ids and m["id"] not in failed_meta:
                        failed_meta[m["id"]] = m

    print(f"  Found meta for {len(failed_meta)}/{len(failed_ids)} failed samples")

    batch_dir = get_batch_dir(args.batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)

    input_path = batch_dir / "input.jsonl"
    meta_path = batch_dir / "meta.jsonl"
    count = 0

    with input_path.open("w", encoding="utf-8") as f_in, \
         meta_path.open("w", encoding="utf-8") as f_meta:

        for sid, meta in sorted(failed_meta.items()):
            options = meta.get("options", {})
            retrieved_docs = normalize_retrieved_docs(meta)
            documents = build_documents(retrieved_docs)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=meta["question"],
                option_a=options.get("A", ""),
                option_b=options.get("B", ""),
                option_c=options.get("C", ""),
                option_d=options.get("D", ""),
                documents=documents,
            )

            custom_id = f"e2e-{sid}"

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                },
            }
            f_in.write(json.dumps(request, ensure_ascii=False) + "\n")
            f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1

    file_size_mb = input_path.stat().st_size / 1024 / 1024
    print(f"[prepare_retry] {count} requests → {input_path} ({file_size_mb:.1f} MB)")


# ── CLI dispatch ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate MedJudgeRAG teacher traces via GPT-5.1 Batch API"
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--batch_id", type=int, default=0)
    shared.add_argument("--model", type=str, default="gpt-5.1")
    shared.add_argument("--train_input", type=str, default=str(TRAIN_INPUT))
    shared.add_argument("--val_input", type=str, default=str(VAL_INPUT))
    shared.add_argument("--offset", type=int, default=0)
    shared.add_argument("--limit", type=int, default=0)

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("prepare", parents=[shared])
    subparsers.add_parser("submit", parents=[shared])
    subparsers.add_parser("status", parents=[shared])
    subparsers.add_parser("download", parents=[shared])
    subparsers.add_parser("merge", parents=[shared])
    subparsers.add_parser("validate", parents=[shared])
    subparsers.add_parser("prepare_retry", parents=[shared])

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "status": cmd_status,
        "download": cmd_download,
        "merge": cmd_merge,
        "validate": cmd_validate,
        "prepare_retry": cmd_prepare_retry,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
