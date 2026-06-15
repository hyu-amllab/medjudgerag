#!/usr/bin/env python3
"""
postprocess_kg_traces.py — Light KG postprocessing for MedJudgeRAG SFT data.

Targets the new format:
  - Entities: ("Entity", <Name>, <Type>, <Description>, <Evidence>)
  - Relations: R1: ("Relation", <Source>, <RelationType>, <Target>, <Description>, <Evidence>)
  - Analysis: KG Entities: [...], KG Relations: [R1, R2]

Actions (light mode):
  1. Remove relations with invalid relation types
  2. Remove relations with invalid/missing entity endpoints (orphans)
  3. Remove entities with invalid entity types (normalize common aliases)
  4. Remove exact duplicate entities (within a sample)
  5. Remove exact duplicate relations (within a sample)
  6. Re-emit KG preserving original R-id (gaps possible: R1, R3, R5 ...)
  7. Drop samples whose answer_target references an R-id/entity that is not in
     the kept KG (covers both newly removed and pre-existing mismatches)

Usage:
    python postprocess_kg_traces.py \
        --input data/teacher_traces_raw.jsonl \
        --output data/teacher_traces_postprocessed.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter


# ── Allowed schema ─────────────────────────────────────────────
ALLOWED_ENTITY_TYPES = {
    "Activities & Behaviors", "Anatomy", "Chemicals & Drugs",
    "Concepts & Ideas", "Devices", "Disorders",
    "Genes & Molecular Sequences", "Geographic Areas",
    "Living Beings", "Objects", "Occupations", "Organizations",
    "Phenomena", "Physiology", "Procedures",
}

# Normalize common aliases → allowed types
ENTITY_TYPE_MAP = {
    "Proteins": "Genes & Molecular Sequences",
    "Enzymes": "Genes & Molecular Sequences",
    "Molecular Sequences": "Genes & Molecular Sequences",
    "Enzymes & Molecular Sequences": "Genes & Molecular Sequences",
    "Cells": "Anatomy",
    "Drugs": "Chemicals & Drugs",
    "Molecules": "Chemicals & Drugs",
    "Substances": "Chemicals & Drugs",
    "Processes": "Phenomena",
    "Events": "Phenomena",
    "Measures": "Phenomena",
}

POSITIVE_RELATION_TYPES = {
    "part_of", "located_in", "connected_to", "adjacent_to", "performs",
    "uses", "affects", "causes", "result_of", "indicates", "measures",
    "diagnoses", "manifestation_of", "precedes", "co_occurs_with",
}


def is_allowed_relation_type(rtype):
    """Accept positive types and canonical negation (not_X)."""
    if rtype in POSITIVE_RELATION_TYPES:
        return True
    if rtype.startswith("not_"):
        base = rtype[4:]
        return base in POSITIVE_RELATION_TYPES
    return False


# ── Parsers ────────────────────────────────────────────────────
_FIELD_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_ENT_LINE_RE = re.compile(r'^\s*\("Entity"')
_REL_LINE_RE = re.compile(r'^\s*(R\d+):\s*\("Relation"')
_REL_LINE_NOID_RE = re.compile(r'^\s*\("Relation"')


def parse_entity(line):
    """Parse ("Entity", name, type, desc, evidence) — returns tuple or None."""
    fields = _FIELD_RE.findall(line)
    # fields[0] = "Entity" marker; data at 1..4
    if len(fields) >= 5:
        return fields[1], fields[2], fields[3], fields[4]
    return None


def parse_relation(line):
    """Parse R_id: ("Relation", src, rtype, tgt, desc, evidence).
    Returns (rid, src, rtype, tgt, desc, evidence) or None.
    """
    rid_m = _REL_LINE_RE.match(line)
    rid = rid_m.group(1) if rid_m else None
    fields = _FIELD_RE.findall(line)
    # fields[0] = "Relation" marker; data at 1..5
    if len(fields) >= 6:
        return rid, fields[1], fields[2], fields[3], fields[4], fields[5]
    return None


def rebuild_entity(name, etype, desc, evidence):
    return f'("Entity", "{name}", "{etype}", "{desc}", "{evidence}")'


def rebuild_relation(rid, src, rtype, tgt, desc, evidence):
    return f'{rid}: ("Relation", "{src}", "{rtype}", "{tgt}", "{desc}", "{evidence}")'


def extract_analysis_refs(at):
    """Extract all KG Entity names and R-ids referenced across all option blocks."""
    ent_refs = set()
    rel_refs = set()
    for opt in ["A", "B", "C", "D"]:
        start_m = re.search(rf'\[{opt}\]', at)
        if not start_m:
            continue
        end_m = re.search(r'(?:\[[ABCD]\]|Decision:)', at[start_m.end():])
        block_end = start_m.end() + end_m.start() if end_m else len(at)
        block = at[start_m.start():block_end]

        kge_m = re.search(r'KG Entities:\s*(\[.*?\])', block, re.DOTALL)
        if kge_m:
            try:
                ents = json.loads(kge_m.group(1))
                if isinstance(ents, list):
                    for e in ents:
                        if isinstance(e, str):
                            ent_refs.add(e)
            except json.JSONDecodeError:
                pass

        kgr_m = re.search(r'KG Relations:\s*(\[.*?\])', block, re.DOTALL)
        if kgr_m:
            for rid in re.findall(r'\bR\d+\b', kgr_m.group(1)):
                rel_refs.add(rid)
    return ent_refs, rel_refs


# ── Postprocess one sample ─────────────────────────────────────
def postprocess_kg(kg_text):
    """Returns (new_kg_text, removed_rids_set, removed_entities_set, stats)."""
    stats = Counter()

    lines = kg_text.split("\n")
    entity_records = []  # list of (name, etype, desc, evidence)
    relation_records = []  # list of (orig_rid, src, rtype, tgt, desc, evidence)

    for line in lines:
        stripped = line.strip()
        if _ENT_LINE_RE.match(stripped):
            parsed = parse_entity(stripped)
            if parsed:
                entity_records.append(parsed)
            else:
                stats["entity_parse_fail"] += 1
        elif _REL_LINE_RE.match(stripped) or _REL_LINE_NOID_RE.match(stripped):
            parsed = parse_relation(stripped)
            if parsed:
                relation_records.append(parsed)
            else:
                stats["relation_parse_fail"] += 1

    # Pass 1: normalize entity types + dedup
    entity_order = []
    entities = {}  # name → (etype, desc, evidence)
    for name, etype, desc, evidence in entity_records:
        # Normalize type
        if etype not in ALLOWED_ENTITY_TYPES:
            if etype in ENTITY_TYPE_MAP:
                etype = ENTITY_TYPE_MAP[etype]
                stats["entity_type_normalized"] += 1
            else:
                # Unknown type → drop
                stats["entity_invalid_type_removed"] += 1
                continue

        if name in entities:
            stats["duplicate_entity_removed"] += 1
            continue
        entities[name] = (etype, desc, evidence)
        entity_order.append(name)

    entity_names = set(entity_order)

    # Pass 2: filter relations
    relation_order = []  # list of (orig_rid, src, rtype, tgt, desc, evidence)
    seen_rel_keys = set()
    removed_rids = set()

    for orig_rid, src, rtype, tgt, desc, evidence in relation_records:
        # Check relation type
        if not is_allowed_relation_type(rtype):
            stats["relation_invalid_type_removed"] += 1
            if orig_rid:
                removed_rids.add(orig_rid)
            continue

        # Check endpoints
        if src not in entity_names or tgt not in entity_names:
            stats["orphan_relation_removed"] += 1
            if orig_rid:
                removed_rids.add(orig_rid)
            continue

        key = (src, rtype, tgt)
        if key in seen_rel_keys:
            stats["duplicate_relation_removed"] += 1
            if orig_rid:
                removed_rids.add(orig_rid)
            continue
        seen_rel_keys.add(key)

        relation_order.append((orig_rid, src, rtype, tgt, desc, evidence))

    # Identify removed entities (for orphan check in answer_target)
    removed_entities = set()
    for name, etype, desc, evidence in entity_records:
        if name not in entity_names:
            removed_entities.add(name)

    # Emit canonical format — preserve original R-id (gaps possible)
    kept_rids = set()
    new_lines = ["Entities:"]
    for name in entity_order:
        etype, desc, evidence = entities[name]
        new_lines.append(rebuild_entity(name, etype, desc, evidence))
    new_lines.append("")
    new_lines.append("Relations:")
    max_fallback = 1
    for orig_rid, src, rtype, tgt, desc, evidence in relation_order:
        if orig_rid:
            rid = orig_rid
        else:
            rid = f"R{max_fallback}"
            max_fallback += 1
        kept_rids.add(rid)
        new_lines.append(rebuild_relation(rid, src, rtype, tgt, desc, evidence))

    return "\n".join(new_lines), kept_rids, entity_names, stats


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Light KG postprocess for MedJudgeRAG data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dropped_output", default=None,
                        help="Optional: file to save dropped samples (for inspection)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_stats = Counter()
    kept = 0
    dropped_dangling = 0
    dropped_items = []

    with in_path.open(encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            kg_text = item.get("kg", "")
            at_text = item.get("answer_target", "")

            if kg_text.strip():
                new_kg, kept_rids, kept_entities, stats = postprocess_kg(kg_text)
                for k, v in stats.items():
                    total_stats[k] += v

                # Verify ALL analysis refs exist in kept KG (not just ones we removed)
                ent_refs, rel_refs = extract_analysis_refs(at_text)
                missing_ents = ent_refs - kept_entities
                missing_rels = rel_refs - kept_rids
                if missing_ents or missing_rels:
                    dropped_dangling += 1
                    dropped_items.append(item["id"])
                    continue

                item["kg"] = new_kg

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    total_stats["kept"] = kept
    total_stats["dropped_dangling"] = dropped_dangling

    print(f"[postprocess_kg_traces] input={in_path}")
    print(f"  Kept: {kept}")
    print(f"  Dropped (dangling refs in answer_target): {dropped_dangling}")
    print(f"  Entity type normalized: {total_stats['entity_type_normalized']}")
    print(f"  Entity invalid type removed: {total_stats['entity_invalid_type_removed']}")
    print(f"  Duplicate entity removed: {total_stats['duplicate_entity_removed']}")
    print(f"  Relation invalid type removed: {total_stats['relation_invalid_type_removed']}")
    print(f"  Orphan relation removed: {total_stats['orphan_relation_removed']}")
    print(f"  Duplicate relation removed: {total_stats['duplicate_relation_removed']}")
    print(f"  Entity parse fail: {total_stats['entity_parse_fail']}")
    print(f"  Relation parse fail: {total_stats['relation_parse_fail']}")
    print(f"  Output: {out_path}")

    if args.dropped_output and dropped_items:
        Path(args.dropped_output).write_text(
            json.dumps(dropped_items, indent=2),
            encoding="utf-8",
        )
        print(f"  Dropped IDs saved: {args.dropped_output}")


if __name__ == "__main__":
    main()
