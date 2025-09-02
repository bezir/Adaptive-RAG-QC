# Adaptive RAG Comprehensive Benchmark

**Comprehensive experiments** across architectural comparison: BERT vs FLAN-T5 vs LLM self-classification.

## Quick Start

```bash
# Run complete architectural comparison
bash scripts/run_experiments_nohup.sh

# Monitor progress
bash scripts/monitor_experiments.sh

# Stop all experiments
bash scripts/monitor_experiments.sh kill
```

## Experiment Overview

**Complete architectural evaluation** across all configurations:
- **Baseline**: Fixed strategy experiments (NOR, ONER, IRCOT)
- **BERT/FLAN-T5**: Traditional classifier experiments  
- **LLM Self-Classification**: Zero-shot classification experiments

**Configuration**:
- **Datasets**: musique, 2wikimultihopqa, trivia, squad, hotpotqa, nq
- **Generators**: gemini-2.5-flash-lite, gemini-1.5-flash-8b
- **Workers**: Parallel processing per experiment
- **Execution**: Sequential (prevents resource conflicts)

## Key Research Results

### Generator Characteristics
- **Gemini-2.5-flash-lite**: Superior parametric knowledge and overall performance
- **Gemini-1.5-flash-8b**: Better at step-by-step reasoning tasks

### Architecture Performance  
- **BERT-Large**: Best downstream performance with fewer parameters
- **FLAN-T5-Large**: Higher parameter count, competitive classification accuracy
- **LLM Self-Classification**: Shows overconfidence, leads to insufficient retrieval

### Dataset Difficulty Hierarchy
1. **Natural Questions** (easiest - factual knowledge)
2. **SQuAD** (reading comprehension)
3. **TriviaQA** (trivia facts)
4. **HotpotQA** (multi-hop reasoning)
5. **2WikiMultiHopQA** (structured reasoning)
6. **MuSiQue** (hardest - complex multi-step)

## Experiment Configuration

Edit `scripts/run_experiments_nohup.sh`:
```bash
WORKERS=N                     # Parallel workers per experiment
MAX_QUERIES=N                # Queries per experiment
GENERATORS=("gemini-2.5-flash-lite" "gemini-1.5-flash-8b")
DATASETS=("musique" "2wikimultihopqa" "trivia" "squad" "hotpotqa" "nq")
```

## Output Structure
```
runs/sequential_experiments_YYYYMMDD_HHMMSS/
├── baseline_*/              # Fixed strategy results
├── adaptive_bert_*/          # BERT classifier results
├── adaptive_flan_t5_*/       # FLAN-T5 classifier results
├── adaptive_llm_*/           # LLM self-classification results
└── aggregated_results.json   # Combined performance metrics
```

## Key Findings

1. **Architecture Efficiency**: BERT-Large outperforms FLAN-T5-Large with fewer parameters
2. **Self-Classification Failure**: LLMs are overconfident, leading to insufficient retrieval
3. **Generator Impact**: Gemini-2.5-flash-lite's strong parametric knowledge improves overall performance
4. **Dataset Complexity**: Clear hierarchy from factual questions to multi-hop reasoning

## Monitoring Commands

```bash
# Status overview
./scripts/monitor_experiments.sh

# Live progress
tail -f runs/sequential_experiments_console.log

# Resource usage
watch -n 1 nvidia-smi

# Kill all experiments
./scripts/monitor_experiments.sh kill
```