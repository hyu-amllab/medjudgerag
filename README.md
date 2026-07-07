# MedJudgeRAG

Official code and sanitized data release for **MedJudgeRAG: Option-Wise Evidence Judgment with Dynamic Knowledge Graphs for Medical MCQA**.

This work is presented at [The Workshop on Graph Foundation Models: A New Era for Graph Machine Learning](https://sites.google.com/view/gfmicml-2026), held at the 43rd International Conference on Machine Learning (ICML 2026), Seoul, South Korea.

## What Is Released

This repository contains:

- MedJudgeRAG training and evaluation code.
- Sanitized teacher traces for SFT.
- Evaluation samples with retrieval document IDs.
- Aggregate result files used for paper tables/analysis.

Raw retrieved document text is **not redistributed** in the public data files. The public files keep retrieval IDs and their order only.

## Setup

Training and evaluation are run in two separate Python environments because vLLM requires a more recent PyTorch than the QLoRA + DeepSpeed training stack.

```bash
# Training environment (data preparation, SFT, KG analysis)
python -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Evaluation environment (vLLM inference)
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

Both environments pin to the exact package versions used in the paper.

## Repository Layout

```text
configs/
  deepspeed_zero2.json

src/
  evaluate.py
  generate_teacher_traces.py
  postprocess_kg_traces.py
  filter_traces_by_length.py
  train_medjudgerag_mistral.py
  train_medjudgerag_llama3.py
  merge_lora_adapter.py
  create_nokg_control_traces.py
  filter_nokg_traces_by_length.py
  train_nokg_control.py
  analyze_kg_quality.py
  plot_kg_quality.py
  rehydrate_retrieved_docs.py

optional/
  build_test_context.py

data/
  medqa_eval_retrieval_ids.jsonl
  medmcqa_eval_retrieval_ids.jsonl
  teacher_traces_postprocessed.jsonl
  teacher_traces_mistral_8192.jsonl
  teacher_traces_llama3_8192.jsonl

results/
  main_results.csv
  explicit_implicit_pairwise.json
  ablation_summary.json
  parametric_explicit_transitions.json
  kg_quality/

requirements-train.txt
requirements-eval.txt
```

## Data Files

To use the dataset on Hugging Face, please visit https://huggingface.co/datasets/youarethewon/medjudgerag.

`data/medqa_eval_retrieval_ids.jsonl` and `data/medmcqa_eval_retrieval_ids.jsonl` contain benchmark questions, answer options, labels, and top-5 retrieved document IDs.
They do not include retrieved passage text.

`data/teacher_traces_postprocessed.jsonl` is the tokenizer-agnostic, sanitized postprocessed teacher trace file.
Use this file to inspect the released teacher targets or to rebuild a full-context trace file for another backbone.

`data/teacher_traces_mistral_8192.jsonl` is the sanitized Mistral-7B-Instruct-v0.3 trace file after chat-template rendering and 8,192-token filtering.

`data/teacher_traces_llama3_8192.jsonl` is the sanitized Meta-Llama-3-8B-Instruct trace file after chat-template rendering and 8,192-token filtering.

The two public file families use slightly different field names for the list of retrieval IDs:

| File family | ID field name |
| --- | --- |
| SFT teacher traces (`teacher_traces_*.jsonl`) | `retrieved_doc_ids` |
| Evaluation context files (`*_eval_retrieval_ids.jsonl`) | `retrieved_ids` |

Evidence markers such as `[1]`, `[2]`, ... refer to the 1-based positions of these IDs in the corresponding list:

```text
[1] = retrieved_doc_ids[0]   (or retrieved_ids[0])
[2] = retrieved_doc_ids[1]   (or retrieved_ids[1])
...
```

To run document-augmented evaluation or to train from prompts that require document text, reconstruct full contexts from the original retrieval corpus/index using these IDs.
`src/rehydrate_retrieved_docs.py` handles both `retrieved_doc_ids` and `retrieved_ids` and writes either `retrieved_docs` (default, for SFT files) or `retrieved_passages` (for evaluation files).
By default, rehydration fails if any retrieval ID cannot be resolved; pass `--allow_missing` only if empty document text is acceptable.
The training and length-filtering scripts require a full-context trace file containing either `retrieved_docs` or `retrieved_passages`; they intentionally abort if only retrieval IDs are provided, to avoid silently training with empty document context.

### Reconstructing Full-Context Files

Once a MedRAG-style corpus directory is available locally, the public files can be rehydrated as follows.

For SFT teacher traces:

```bash
python src/rehydrate_retrieved_docs.py \
  --input  data/teacher_traces_mistral_8192.jsonl \
  --output data/full_teacher_traces_mistral_8192.jsonl \
  --medrag_corpus_dir /path/to/medrag/corpus

python src/rehydrate_retrieved_docs.py \
  --input  data/teacher_traces_llama3_8192.jsonl \
  --output data/full_teacher_traces_llama3_8192.jsonl \
  --medrag_corpus_dir /path/to/medrag/corpus
```

For evaluation context files (note the `--output_key`):

```bash
python src/rehydrate_retrieved_docs.py \
  --input  data/medqa_eval_retrieval_ids.jsonl \
  --output data/full_context_medqa.jsonl \
  --medrag_corpus_dir /path/to/medrag/corpus \
  --output_key retrieved_passages

python src/rehydrate_retrieved_docs.py \
  --input  data/medmcqa_eval_retrieval_ids.jsonl \
  --output data/full_context_medmcqa.jsonl \
  --medrag_corpus_dir /path/to/medrag/corpus \
  --output_key retrieved_passages
```

## Retrieval Setting

The paper uses MedRAG-style retrieval with:

```text
Retriever: Contriever
Corpus: PubMed + medical textbooks
Top-k: 5
```

`optional/build_test_context.py` is included for provenance, but it requires the corresponding corpus/index and is not needed for using the sanitized release files.

## Training

The released `teacher_traces_*.jsonl` files are sanitized and do not include raw retrieved document text.
Before training, reconstruct full-context versions from the retrieval corpus/index, preserving the same sample order and adding `retrieved_docs` or `retrieved_passages`.
The commands below assume the following reconstructed files:

```text
data/full_teacher_traces_mistral_8192.jsonl
data/full_teacher_traces_llama3_8192.jsonl
```

### Mistral

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  src/train_medjudgerag_mistral.py \
  --deepspeed configs/deepspeed_zero2.json \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --train_data data/full_teacher_traces_mistral_8192.jsonl \
  --val_data data/full_teacher_traces_mistral_8192.jsonl \
  --epochs 3 \
  --batch 1 \
  --grad_accum 16 \
  --max_seq_len 8192 \
  --kg_weight 0.0 \
  --analysis_weight 1.0 \
  --output_dir outputs/medjudgerag_mistral_lora
```

### Llama 3

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  src/train_medjudgerag_llama3.py \
  --deepspeed configs/deepspeed_zero2.json \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --train_data data/full_teacher_traces_llama3_8192.jsonl \
  --val_data data/full_teacher_traces_llama3_8192.jsonl \
  --epochs 3 \
  --batch 1 \
  --grad_accum 16 \
  --max_seq_len 8192 \
  --kg_weight 0.0 \
  --analysis_weight 1.0 \
  --output_dir outputs/medjudgerag_llama3_lora
```

GPU IDs, number of processes, and vLLM memory utilization are machine-specific and can be changed through command-line arguments or environment variables.

## Merge LoRA Adapter

```bash
CUDA_VISIBLE_DEVICES=0 python src/merge_lora_adapter.py \
  --gpu 0 \
  --base_model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path outputs/medjudgerag_mistral_lora \
  --output_dir outputs/merged_mistral_medjudgerag
```

## Evaluation

`evaluate.py` supports the official mode names:

```text
parametric   Q + options only
vanilla_rag  Q + options + documents using the MedRAG prompt
explicit     MedJudgeRAG explicit KG generation + option-wise reasoning
implicit     MedJudgeRAG option-wise reasoning without explicit KG generation
```

Legacy aliases are also accepted internally:

```text
naive_rag = vanilla_rag
stage2 = explicit
analysis_only = implicit
```

Example with a reconstructed full-context file:

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
  --model_path outputs/merged_mistral_medjudgerag \
  --mode explicit \
  --benchmark medqa \
  --data_path data/full_context_medqa.jsonl \
  --tag mistral_explicit_medqa \
  --vllm \
  --max_new_tokens 8192
```

The public `*_retrieval_ids.jsonl` files do not contain document text. Therefore `vanilla_rag`, `explicit`, and `implicit` evaluation require a full-context JSONL containing `retrieved_passages`.

## No-KG Control

The no-KG control data can be regenerated from full-context teacher traces.

```bash
python src/create_nokg_control_traces.py \
  --input data/full_teacher_traces_mistral_8192.jsonl \
  --output data/full_nokg_control_mistral_8192.jsonl
```

For Llama 3, use the Llama-filtered traces instead:

```bash
python src/create_nokg_control_traces.py \
  --input data/full_teacher_traces_llama3_8192.jsonl \
  --output data/full_nokg_control_llama3_8192.jsonl
```

Then train Mistral:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  src/train_nokg_control.py \
  --deepspeed configs/deepspeed_zero2.json \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --train_data data/full_nokg_control_mistral_8192.jsonl \
  --val_data data/full_nokg_control_mistral_8192.jsonl \
  --epochs 3 \
  --batch 1 \
  --grad_accum 16 \
  --max_seq_len 8192 \
  --kg_weight 0.0 \
  --analysis_weight 1.0 \
  --output_dir outputs/nokg_control_mistral_lora
```

For Llama 3, pass `--model_name meta-llama/Meta-Llama-3-8B-Instruct`,
`--train_data data/full_nokg_control_llama3_8192.jsonl`, and
`--val_data data/full_nokg_control_llama3_8192.jsonl`.

## Teacher Trace Generation

`src/generate_teacher_traces.py` contains the teacher prompt and batch workflow used to generate MedJudgeRAG traces.
The gold answer is not included in the teacher prompt and is used only for post-hoc validation.

## Reproducing the Result Files

The aggregate files in `results/` are produced by the analysis scripts in `src/`.
The general flow is:

1. Run `src/evaluate.py` for every (backbone, mode, benchmark, λ_g) combination and write the per-sample predictions to a JSONL file under `outputs/eval_*.jsonl`.
2. Aggregate accuracies into `results/main_results.csv` from those JSONL files.
3. Compute KG-quality statistics with `src/analyze_kg_quality.py`, writing one JSON file per (backbone, benchmark) under `results/kg_quality/`.
4. Render Figure 2 with `src/plot_kg_quality.py`, which reads the JSON files in `results/kg_quality/` and writes PDFs/PNGs into `results/kg_quality/figures/`.

KG-quality analysis (per backbone × benchmark):

```bash
python src/analyze_kg_quality.py \
  --auto_discover \
  --backbone mistral --benchmark medqa \
  --eval_dir outputs \
  --output results/kg_quality/mistral_medqa

python src/analyze_kg_quality.py \
  --auto_discover \
  --backbone mistral --benchmark medmcqa \
  --eval_dir outputs \
  --output results/kg_quality/mistral_medmcqa

python src/analyze_kg_quality.py \
  --auto_discover \
  --backbone llama3 --benchmark medqa \
  --eval_dir outputs \
  --output results/kg_quality/llama3_medqa

python src/analyze_kg_quality.py \
  --auto_discover \
  --backbone llama3 --benchmark medmcqa \
  --eval_dir outputs \
  --output results/kg_quality/llama3_medmcqa
```

Figure 2 (combined 4-panel figure with shared legend):

```bash
python src/plot_kg_quality.py
```

The script reads the four JSON files above from `results/kg_quality/` and writes the combined figure to `results/kg_quality/figures/`.

## Citation

If you use this code or the released MedJudgeRAG teacher traces, please cite the paper.
The entry below is a temporary pre-camera-ready citation and will be updated once the final camera-ready metadata is available.

```bibtex
@inproceedings{
seo2026medjudgerag,
title={MedJudge{RAG}: Option-Wise Evidence Judgment with Dynamic Knowledge Graphs for Medical {MCQA}},
author={Seongwon Seo and Seung Hwan Cho and Young-Min Kim},
booktitle={Workshop on Graph Foundation Models: A New Era for Graph Machine Learning},
year={2026},
url={https://openreview.net/forum?id=lrZIKfoRfz}
}
```

## Acknowledgements and Upstream Resources

This project builds on several public medical QA and retrieval resources. Please cite and follow the licenses/terms of the original resources when using this repository.

- [MedQA / USMLE 4-option dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), derived from the MedQA benchmark introduced in [*What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams*](https://arxiv.org/abs/2009.13081).
- [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), introduced in [*MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering*](https://arxiv.org/abs/2203.14371).
- [MedRAG](https://github.com/gzxiong/MedRAG), the medical RAG toolkit and corpus/indexing setup used as the retrieval basis, introduced in [*Benchmarking Retrieval-Augmented Generation for Medicine*](https://arxiv.org/abs/2402.13178).

Our released data keeps only retrieval document IDs and does not redistribute the raw MedRAG corpus passages.

## License and Data Notice

MedQA, MedMCQA, PubMed, and textbook-derived corpora have their own licenses and citation requirements.
This release avoids redistributing raw retrieved document text and provides retrieval IDs instead.
Users are responsible for complying with the terms of the original datasets and corpora.
