# Optimized Labeling System

## Quick Start

### Gemini Models (Recommended)
```bash
export GOOGLE_API_KEY="your-api-key"

# Optimized strategy (significantly fewer steps)
bash shell_scripts/run_master_labeling.sh \
  --model gemini-2.5-flash-lite \
  --strategy optimized \
  --size 3000 \
  --dataset all

# Original strategy (for comparison)
bash shell_scripts/run_master_labeling.sh \
  --model gemini-2.5-flash-lite \
  --strategy original \
  --size 1000 \
  --dataset hotpotqa
```

## Key Research Contribution

### Optimized Strategy Logic
- **Single-hop datasets** (NQ, SQuAD, TriviaQA): Run NOR only
  - NOR success → 'A', NOR failure → 'B' (no ONER/IRCOT needed)
- **Multi-hop datasets** (HotpotQA, 2Wiki, MuSiQue): Run NOR → ONER
  - NOR success → 'A', ONER success → 'B', both fail → 'C' (no IRCOT needed)

### Efficiency Gains
- **Single-hop datasets**: Eliminates unnecessary ONER and IRCOT steps
- **Multi-hop datasets**: Avoids expensive IRCOT execution when simpler methods work
- **Overall**: Dramatic reduction in computational requirements

## Convert to Classifier Format

```bash
# Create balanced classifier training data
python scripts/convert_to_classifier_format.py \
  --input_folder predictions \
  --output_dir ../classifier/data \
  --model_name gemini-2.5-flash-lite \
  --dataset_name optimized_silver \
  --validation_ratio 0.2 \
  --multiple
```

## Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--strategy` | optimized, original | optimized | Labeling strategy |
| `--model` | gemini-2.5-flash-lite, gemini-1.5-flash-8b | required | Model name |
| `--dataset` | all, hotpotqa, musique, etc | required | Dataset selection |
| `--size` | int | 1000 | Samples per dataset |
| `--workers` | int | varies by model type | Parallel workers |

## Output Structure
```
predictions/
├── dev_3000/
│   ├── optimized_strategy/
│   │   └── {dataset}_{model}_optimized_3000_results.json
│   └── original_strategy/
│       └── {dataset}_{model}_original_1000_results.json
```
