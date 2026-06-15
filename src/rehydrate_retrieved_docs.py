#!/usr/bin/env python3
"""
rehydrate_retrieved_docs.py

Reconstruct full retrieved-document text in the sanitized release files
by looking each ID up in a local MedRAG-style corpus.

The public release stores only document identifiers under either
``retrieved_doc_ids`` (SFT teacher traces) or ``retrieved_ids``
(evaluation context files). Training and document-augmented evaluation
require the actual passage text. This script reads the IDs, fetches
each entry from a MedRAG-format corpus directory, and writes a new
JSONL file in which the IDs are expanded into full ``retrieved_docs``
records (or plain-text ``retrieved_passages`` for evaluation files).

Expected corpus layout
----------------------
The MedRAG corpus is a directory containing one or more sub-corpora
(e.g. ``pubmed``, ``textbooks``, ``statpearls``). Each sub-corpus
holds JSONL chunk files under ``chunk/`` where every line is of the
form::

    {"id": "<chunk_id>", "title": "...", "content": "...", ...}

The script scans every ``*.jsonl`` file under the corpus directory
(recursively) and indexes records by their ``id`` field.

Usage
-----
Rehydrate a teacher-trace SFT file (writes ``retrieved_docs``)::

    python src/rehydrate_retrieved_docs.py \\
        --input  data/teacher_traces_mistral_8192.jsonl \\
        --output data/full_teacher_traces_mistral_8192.jsonl \\
        --medrag_corpus_dir /path/to/medrag/corpus

Rehydrate an evaluation context file (writes ``retrieved_passages``)::

    python src/rehydrate_retrieved_docs.py \\
        --input  data/medqa_eval_retrieval_ids.jsonl \\
        --output data/full_context_medqa.jsonl \\
        --medrag_corpus_dir /path/to/medrag/corpus \\
        --output_key retrieved_passages
"""

import argparse
import json
import os
import sys
from pathlib import Path


# JSON keys that may carry the list of retrieved document IDs.
ID_KEYS = ("retrieved_doc_ids", "retrieved_ids")


def collect_needed_ids(input_path):
    """First pass: scan the input file and collect every unique
    retrieval ID that must be resolved. Returns a set of strings.

    Reading the input first lets us build a smaller in-memory corpus
    index in the second pass instead of loading the entire MedRAG
    corpus.
    """
    needed = set()
    n_lines = 0
    n_items_with_ids = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            item = json.loads(line)
            ids = None
            for key in ID_KEYS:
                if key in item and item[key] is not None:
                    ids = item[key]
                    break
            if ids is None:
                continue
            n_items_with_ids += 1
            for doc_id in ids:
                if isinstance(doc_id, str):
                    needed.add(doc_id)
    print(f"[scan] {input_path}: {n_lines} lines, "
          f"{n_items_with_ids} with retrieval IDs, "
          f"{len(needed)} unique IDs to resolve",
          file=sys.stderr)
    return needed


def build_corpus_index(corpus_dir, needed_ids):
    """Second pass: walk the corpus directory and build an in-memory
    index limited to the IDs that the input actually needs.

    Returns a dict::

        {id: {"title": str, "content": str}}
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    index = {}
    n_files = 0
    n_records = 0
    for jsonl_path in sorted(corpus_dir.rglob("*.jsonl")):
        n_files += 1
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    n_records += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = rec.get("id")
                    if not isinstance(doc_id, str) or doc_id not in needed_ids:
                        continue
                    # Prefer the explicit `content` field; fall back to
                    # `contents` (used by some MedRAG sub-corpora).
                    content = rec.get("content")
                    if content is None:
                        content = rec.get("contents", "")
                    index[doc_id] = {
                        "title": rec.get("title", ""),
                        "content": content,
                    }
                    if len(index) == len(needed_ids):
                        # Found everything; stop early.
                        print(
                            f"[index] resolved all {len(needed_ids)} IDs "
                            f"after scanning {n_files} files, "
                            f"{n_records} records",
                            file=sys.stderr,
                        )
                        return index
        except OSError as exc:
            print(f"[warn] skipping {jsonl_path}: {exc}", file=sys.stderr)

    missing = needed_ids - set(index.keys())
    print(
        f"[index] scanned {n_files} files, {n_records} records; "
        f"resolved {len(index)}/{len(needed_ids)} IDs "
        f"(missing: {len(missing)})",
        file=sys.stderr,
    )
    if missing:
        sample = sorted(missing)[:5]
        print(
            f"[warn] {len(missing)} IDs not found in corpus. Examples: {sample}",
            file=sys.stderr,
        )
    return index


def format_passage(entry):
    """Format one corpus record as the plain passage string expected by evaluate.py."""
    title = (entry.get("title") or "").strip()
    content = (entry.get("content") or "").strip()
    if title and content:
        return f"{title}. {content}"
    return title or content


def rehydrate(input_path, output_path, index, output_key):
    """Third pass: rewrite the input file with full retrieved-document
    records inserted under ``output_key`` while preserving the
    original ID order.
    """
    n_in = 0
    n_out = 0
    n_missing_per_item = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            item = json.loads(line)

            ids = None
            for key in ID_KEYS:
                if key in item and item[key] is not None:
                    ids = item[key]
                    break

            if ids is None:
                # Nothing to rehydrate; pass through unchanged.
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_out += 1
                continue

            payload = []
            for doc_id in ids:
                entry = index.get(doc_id, {})
                if output_key == "retrieved_passages":
                    payload.append(format_passage(entry))
                else:
                    payload.append({
                        "id": doc_id,
                        "title": entry.get("title", ""),
                        "content": entry.get("content", ""),
                    })
                if doc_id not in index:
                    n_missing_per_item += 1

            item[output_key] = payload
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"[write] {output_path}: {n_out} lines written "
        f"(input: {n_in}, missing IDs in payload: {n_missing_per_item})",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct full retrieved-document text in a sanitized "
            "MedJudgeRAG JSONL file using a MedRAG-style corpus."
        )
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a sanitized JSONL file with retrieval IDs.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the rehydrated JSONL file.",
    )
    parser.add_argument(
        "--medrag_corpus_dir", required=True,
        help=(
            "Path to a MedRAG-style corpus directory containing JSONL "
            "chunk files. The directory is scanned recursively for "
            "'*.jsonl' files."
        ),
    )
    parser.add_argument(
        "--output_key", default="retrieved_docs",
        choices=["retrieved_docs", "retrieved_passages"],
        help=(
            "Name of the output field that will hold the full document "
            "records. Use 'retrieved_docs' for SFT teacher traces and "
            "'retrieved_passages' for evaluation context files."
        ),
    )
    parser.add_argument(
        "--allow_missing", action="store_true",
        help=(
            "Allow unresolved retrieval IDs and fill them with empty text. "
            "By default, unresolved IDs are treated as an error to avoid "
            "silently training/evaluating with missing document context."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")

    needed = collect_needed_ids(args.input)
    if not needed:
        print(
            "[warn] no retrieval IDs found in input; copying through.",
            file=sys.stderr,
        )
        # Copy through unchanged.
        with open(args.input, "r", encoding="utf-8") as fin, \
             open(args.output, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)
        return

    index = build_corpus_index(args.medrag_corpus_dir, needed)
    missing = needed - set(index.keys())
    if missing and not args.allow_missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            f"Failed to resolve {len(missing)} retrieval IDs from "
            f"{args.medrag_corpus_dir}. Examples: {sample}. "
            "Pass --allow_missing only if empty document text is acceptable."
        )
    rehydrate(args.input, args.output, index, args.output_key)


if __name__ == "__main__":
    main()
