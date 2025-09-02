# Multi-Architecture Classifier Training

**Key Finding**: BERT-Large achieves competitive performance with **significantly fewer parameters** than FLAN-T5-Large.

## Quick Start

```bash
# Train all architectures (BERT + FLAN-T5, optimized + original strategies)
bash run/train_master.sh

# Or train specific configuration
./run/train_master.sh --model bert --llm gemini-2.5-flash-lite --gpu 3 --epochs 3
./run/train_master.sh --model t5 --llm gemini-2.5-flash-lite --gpu 7 --epochs 20
```

## Research Results

### Architecture Findings
- **BERT-Large**: Achieves competitive performance with fewer parameters
- **FLAN-T5-Large**: Higher parameter count but similar downstream performance
- **Parameter Efficiency**: Encoder-only architectures more suitable for classification tasks

### Strategy Impact
- **Optimized Strategy**: Consistently outperforms original labeling approach
- **Training Efficiency**: Better label quality leads to improved classifier performance
- **Cross-Model Validation**: Benefits observed across different generator models

## Data Preparation

```bash
# Convert silver labeling results to classifier format
python ../scaled_silver_labeling/scripts/convert_to_classifier_format.py \
  --input_folder ../scaled_silver_labeling/predictions \
  --output_dir . \
  --model_name gemini-2.5-flash-lite \
  --dataset_name optimized_silver \
  --validation_ratio 0.2 \
  --multiple
```

## Manual Training Commands

### BERT (Recommended)
```bash
# Optimized labels
python train_classifier.py \
  --architecture bert \
  --model_name bert-large-uncased \
  --train_data data/optimized_silver/train.json \
  --val_data data/optimized_silver/val.json \
  --output_dir models/bert_classifier_optimized \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --epochs 3
```

### FLAN-T5 (Comparison)
```bash
# Optimized labels
python train_classifier.py \
  --architecture flan_t5 \
  --model_name google/flan-t5-large \
  --train_data data/optimized_silver/train.json \
  --val_data data/optimized_silver/val.json \
  --output_dir models/flan_t5_classifier_optimized \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epochs 20
```

## Parameters

| Parameter | BERT | FLAN-T5 | Description |
|-----------|------|---------|-------------|
| `--model` | bert | t5 | Architecture type |
| `--llm` | gemini-2.5-flash-lite, gemini-1.5-flash-8b | Data source |
| `--gpu` | 0-N | GPU device |
| `--epochs` | fewer | more | Training epochs |
| `--batch-size` | auto | auto | Batch size |
| `--learning-rate` | auto | auto | Learning rate |

## Output Structure
```
outputs/
├── bert-large/
│   └── gemini-2.5-flash-lite/
│       └── optimized_5000/
│           ├── best_model.pt
│           ├── validation/
│           └── training_logs.json
└── t5-large/
    └── gemini-2.5-flash-lite/
        └── optimized_5000/
            ├── best_model.pt
            ├── validation/
            └── training_logs.json
```

## Key Insights

1. **Parameter Efficiency**: BERT-Large matches FLAN-T5-Large performance with fewer parameters
2. **Strategy Impact**: Optimized labeling significantly improves classifier accuracy
3. **Architecture Alignment**: Encoder-only models excel at classification tasks
4. **Training Efficiency**: BERT converges faster than FLAN-T5 during training
