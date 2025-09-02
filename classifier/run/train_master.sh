#!/bin/bash

# Master Training Script for BERT and T5 Classifiers
# Usage: ./train_master.sh [OPTIONS]
# Example: ./train_master.sh --model bert --llm gemini-2.5-flash-lite --gpu 0 --epochs 3

set -e  # Exit on any error

# Set up environment variables
export PROJECT_ROOT=${PROJECT_ROOT:-""}

# Default values
MODEL_TYPE="bert"           # bert or t5
LLM_NAME="gemini-2.5-flash-lite"  # gemini-2.5-flash-lite, gemini-1.5-flash-8b, qwen, xl, xxl
GPU=0
EPOCHS=3
DATASET_SIZE=5000
ANNOTATION_TYPE="original"
TYPE="dev"
BATCH_SIZE_OVERRIDE=""
LEARNING_RATE_OVERRIDE=""
DATA_DIR_OVERRIDE=""
ACTUAL_DATASET_SIZE=""

# Help function
show_help() {
    cat << EOF
Master Training Script for BERT and T5 Classifiers

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --model MODEL_TYPE          Model type: bert or t5 (default: bert)
    --llm LLM_NAME             LLM dataset: gemini-2.5-flash-lite, gemini-1.5-flash-8b, qwen, xl, xxl (default: gemini-2.5-flash-lite)
    --gpu GPU_ID               GPU ID to use (default: 0)
    --epochs NUM               Number of epochs (default: 3)
    --dataset-size SIZE        Dataset size (default: 5000)
    --annotation-type TYPE     Annotation type: original, optimized, binary_silver, train_data (default: original)
    --type TYPE                Type: dev, train_data, data (default: dev)
    --batch-size SIZE          Override batch size
    --learning-rate RATE       Override learning rate
    --data-dir PATH            Override data directory path (bypasses automatic path construction)
    --actual-size SIZE         Actual dataset size (auto-detected from folder name if using --data-dir)
    --help                     Show this help message

NOTE: Validation monitoring is always enabled during training (epoch-by-epoch validation)

EXAMPLES:
    # BERT with Gemini 2.5 Flash Lite dataset
    $0 --model bert --llm gemini-2.5-flash-lite --gpu 3 --epochs 3 --dataset-size 5000  --annotation-type original

    # BERT with Gemini 1.5 Flash 8B dataset  
    $0 --model bert --llm gemini-1.5-flash-8b --gpu 1 --epochs 3 --dataset-size 5000 --annotation-type optimized

    # T5 with Gemini 2.5 Flash Lite dataset
    $0 --model t5 --llm gemini-2.5-flash-lite --gpu 7 --epochs 20 --dataset-size 5000  --annotation-type optimized



    # BERT with XL dataset
    $0 --model bert --llm xl --gpu 9 --epochs 3 --dataset-size full_silver_labeled --annotation-type train_data

    # T5 with Qwen dataset
    $0 --model t5 --llm qwen --gpu 7 --epochs 20 --dataset-size musique_hotpot_wiki2_nq_tqa_sqd  --annotation-type optimized

    # Using custom data directory (new format from convert_to_classifier_format.py)
    $0 --model bert --gpu 0 --epochs 3 --data-dir ${PROJECT_ROOT}/Adaptive-RAG/classifier/data/2634_gemini-2.5-flash-lite_optimized
    
    # BERT with 50% dataset size (auto-detects size from folder name)
    $0 --model bert --gpu 1 --epochs 3 --data-dir ${PROJECT_ROOT}/Adaptive-RAG/classifier/data/13174_gemini-2.5-flash-lite_optimized

    # T5 with 100% dataset size (manually specify actual size)
    $0 --model t5 --gpu 2 --epochs 20 --data-dir ${PROJECT_ROOT}/Adaptive-RAG/classifier/data/26349_gemini-2.5-flash-lite_optimized --actual-size 26349

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --llm)
            LLM_NAME="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --dataset-size)
            DATASET_SIZE="$2"
            shift 2
            ;;
        --annotation-type)
            ANNOTATION_TYPE="$2"
            shift 2
            ;;
        --type)
            TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE_OVERRIDE="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER_OVERRIDE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --actual-size)
            ACTUAL_DATASET_SIZE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ "$MODEL_TYPE" != "bert" && "$MODEL_TYPE" != "t5" ]]; then
    echo "Error: Model type must be 'bert' or 't5'"
    exit 1
fi

if [[ "$LLM_NAME" != "gemini-2.5-flash-lite" && "$LLM_NAME" != "gemini-1.5-flash-8b" && "$LLM_NAME" != "qwen" && "$LLM_NAME" != "xl" && "$LLM_NAME" != "xxl" ]]; then
    echo "Error: LLM name must be 'gemini-2.5-flash-lite', 'gemini-1.5-flash-8b', 'qwen', 'xl', or 'xxl'"
    exit 1
fi

# Set model-specific configurations
if [[ "$MODEL_TYPE" == "bert" ]]; then
    CLASSIFICATION_MODEL="google-bert/bert-large-uncased"
    PYTHON_SCRIPT="../run_classifier.py"
else
    CLASSIFICATION_MODEL="t5-large"
    PYTHON_SCRIPT="../run_classifier.py"
fi

# Set LLM-specific configurations
case "$LLM_NAME" in
    "gemini-2.5-flash-lite")
        ACTUAL_LLM_NAME="gemini"
        ;;
    "gemini-1.5-flash-8b")
        ACTUAL_LLM_NAME="gemini"
        ;;
    "qwen")
        ACTUAL_LLM_NAME="qwen"
        ;;
    "xl")
        ACTUAL_LLM_NAME="flan-t5-xl"
        ;;
    "xxl")
        ACTUAL_LLM_NAME="flan-t5-xxl"
        ;;
esac

# Set default training parameters based on model and ACTUAL dataset size
set_default_params() {
    # Use ACTUAL_DATASET_SIZE for parameter configuration
    local dataset_size_for_params="${ACTUAL_DATASET_SIZE:-$DATASET_SIZE}"
    
    # Convert to integer for comparison
    dataset_size_num=$(echo "$dataset_size_for_params" | sed 's/[^0-9]//g')
    
    # Set parameters based on actual dataset size
    if [[ $dataset_size_num -gt 15000 ]]; then
        # Above 15k total samples
        if [[ "$MODEL_TYPE" == "t5" ]]; then
            DEFAULT_TRAIN_BATCH_SIZE=32
            DEFAULT_EVAL_BATCH_SIZE=32
        else
            DEFAULT_TRAIN_BATCH_SIZE=16
            DEFAULT_EVAL_BATCH_SIZE=16
        fi
        DEFAULT_LEARNING_RATE="2e-5"
        DEFAULT_WEIGHT_DECAY="0.01"
        DEFAULT_WARMUP_STEPS=500
    elif [[ $dataset_size_num -gt 5000 ]]; then
        # Above 5k total samples
        if [[ "$MODEL_TYPE" == "t5" ]]; then
            DEFAULT_TRAIN_BATCH_SIZE=32
            DEFAULT_EVAL_BATCH_SIZE=32
        else
            DEFAULT_TRAIN_BATCH_SIZE=8
            DEFAULT_EVAL_BATCH_SIZE=8
        fi
        DEFAULT_LEARNING_RATE="2e-5"
        DEFAULT_WEIGHT_DECAY="0.01"
        DEFAULT_WARMUP_STEPS=300
    elif [[ $dataset_size_num -gt 500 ]]; then
        # Above 500 total samples
        if [[ "$MODEL_TYPE" == "t5" ]]; then
            DEFAULT_TRAIN_BATCH_SIZE=8
            DEFAULT_EVAL_BATCH_SIZE=8
        else
            DEFAULT_TRAIN_BATCH_SIZE=4
            DEFAULT_EVAL_BATCH_SIZE=4
        fi
        DEFAULT_LEARNING_RATE="2e-5"
        DEFAULT_WEIGHT_DECAY="0.01"
        DEFAULT_WARMUP_STEPS=100
    else
        # Very small datasets
        if [[ "$MODEL_TYPE" == "t5" ]]; then
            DEFAULT_TRAIN_BATCH_SIZE=8
            DEFAULT_EVAL_BATCH_SIZE=8
        else
            DEFAULT_TRAIN_BATCH_SIZE=4
            DEFAULT_EVAL_BATCH_SIZE=4
        fi
        DEFAULT_LEARNING_RATE="2e-5"
        DEFAULT_WEIGHT_DECAY="0.01"
        DEFAULT_WARMUP_STEPS=50
    fi
    
    # T5 models get slightly higher learning rate
    if [[ "$MODEL_TYPE" == "t5" ]]; then
        DEFAULT_LEARNING_RATE="3e-5"
    fi
    

}

# Generate timestamp
DATE=$(date +%Y-%m-%d_%H-%M-%S)

# Construct paths based on LLM and dataset configuration
construct_paths() {
    # Check if data directory is overridden
    if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
        DATA_DIR="$DATA_DIR_OVERRIDE"
        DATASET_FOLDER_NAME=$(basename "$DATA_DIR")
        
        # Auto-detect dataset size from folder name if not provided
        if [[ -z "$ACTUAL_DATASET_SIZE" ]]; then
            # Extract size from folder name pattern: SIZE_model_type
            ACTUAL_DATASET_SIZE=$(echo "$DATASET_FOLDER_NAME" | cut -d'_' -f1)
        fi
        
        # Create output dir name (folder name already contains size)
        OUTPUT_DIR="../outputs/${MODEL_TYPE}-large/${DATASET_FOLDER_NAME}_${EPOCHS}ep"
    elif [[ ("$LLM_NAME" == "gemini-2.5-flash-lite" || "$LLM_NAME" == "gemini-1.5-flash-8b") && "$TYPE" == "dev" ]]; then
        # Gemini dev dataset
        DATA_DIR="${PROJECT_ROOT}/Adaptive-RAG/classifier/data/${TYPE}_${DATASET_SIZE}_${ANNOTATION_TYPE}_${ACTUAL_LLM_NAME}"
        if [[ -z "$ACTUAL_DATASET_SIZE" ]]; then
            ACTUAL_DATASET_SIZE="$DATASET_SIZE"
        fi
        OUTPUT_DIR="../outputs/${MODEL_TYPE}-large/${ACTUAL_LLM_NAME}/${ACTUAL_DATASET_SIZE}_${TYPE}_${EPOCHS}ep_${ANNOTATION_TYPE}"
    elif [[ "$LLM_NAME" == "qwen" ]]; then
        # Qwen dataset
        DATA_DIR="../data/${DATASET_SIZE}/${ACTUAL_LLM_NAME}/${ANNOTATION_TYPE}"
        VALID_DATA_DIR="../data/${DATASET_SIZE}/${ACTUAL_LLM_NAME}/silver"
        if [[ -z "$ACTUAL_DATASET_SIZE" ]]; then
            ACTUAL_DATASET_SIZE="$DATASET_SIZE"
        fi
        OUTPUT_DIR="../outputs/${MODEL_TYPE}-large/${ACTUAL_LLM_NAME}/${ACTUAL_DATASET_SIZE}_${EPOCHS}ep_${DATE}"
    else
        # XL/XXL datasets
        DATA_DIR="../data/${DATASET_SIZE}/${ACTUAL_LLM_NAME}"
        if [[ -z "$ACTUAL_DATASET_SIZE" ]]; then
            ACTUAL_DATASET_SIZE="$DATASET_SIZE"
        fi
        OUTPUT_DIR="../outputs/${MODEL_TYPE}-large/${ACTUAL_LLM_NAME}/${ACTUAL_DATASET_SIZE}_${EPOCHS}ep_${DATE}"
    fi
}

construct_paths

# Set parameters based on actual dataset size (now that ACTUAL_DATASET_SIZE is available)
set_default_params

# Apply overrides if provided
TRAIN_BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-$DEFAULT_TRAIN_BATCH_SIZE}
EVAL_BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-$DEFAULT_EVAL_BATCH_SIZE}
LEARNING_RATE=${LEARNING_RATE_OVERRIDE:-$DEFAULT_LEARNING_RATE}
WEIGHT_DECAY=${DEFAULT_WEIGHT_DECAY}
WARMUP_STEPS=${DEFAULT_WARMUP_STEPS}

# Display configuration
echo "=========================================="
echo "Master Training Script Configuration"
echo "=========================================="
echo "Model Type: ${MODEL_TYPE}"
echo "Classification Model: ${CLASSIFICATION_MODEL}"
echo "LLM Name: ${ACTUAL_LLM_NAME}"
echo "GPU: ${GPU}"
echo "Epochs: ${EPOCHS}"
echo "Configured Dataset Size: ${DATASET_SIZE}"
echo "Actual Dataset Size: ${ACTUAL_DATASET_SIZE}"
echo "Annotation Type: ${ANNOTATION_TYPE}"
echo "Type: ${TYPE}"
echo "Validation: Always enabled (epoch-by-epoch)"
echo "Train Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Eval Batch Size: ${EVAL_BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Date: ${DATE}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build base training arguments
build_training_args() {
    TRAINING_ARGS=(
        "--model_name_or_path" "${CLASSIFICATION_MODEL}"
        "--train_file" "${DATA_DIR}/train.json"
        "--question_column" "question"
        "--answer_column" "answer"
        "--learning_rate" "${LEARNING_RATE}"
        "--weight_decay" "${WEIGHT_DECAY}"
        "--max_seq_length" "384"
        "--per_device_train_batch_size" "${TRAIN_BATCH_SIZE}"
        "--per_device_eval_batch_size" "${EVAL_BATCH_SIZE}"
        "--num_train_epochs" "${EPOCHS}"
        "--num_warmup_steps" "${WARMUP_STEPS}"
        "--output_dir" "${OUTPUT_DIR}"
        "--overwrite_cache"
        "--do_train"
        "--seed" "42"
    )

    # Add model-specific arguments
    if [[ "$MODEL_TYPE" == "t5" ]]; then
        TRAINING_ARGS+=("--source_prefix" "classify: ")
    fi

    # Add BERT-specific arguments for certain configurations
    if [[ "$MODEL_TYPE" == "bert" && ("$LLM_NAME" == "xl" || "$LLM_NAME" == "xxl") ]]; then
        TRAINING_ARGS+=("--cache_dir" "../../.cache")
        TRAINING_ARGS+=("--doc_stride" "128")
        TRAINING_ARGS+=("--gradient_accumulation_steps" "2")
        TRAINING_ARGS+=("--train_column" "train")
    fi

    # Add T5-specific arguments for certain configurations
    if [[ "$MODEL_TYPE" == "t5" && "$LLM_NAME" != "gemini-2.5-flash-lite" && "$LLM_NAME" != "gemini-1.5-flash-8b" ]]; then
        TRAINING_ARGS+=("--doc_stride" "128")
        TRAINING_ARGS+=("--train_column" "train")
    fi



    # Always add validation (epoch-by-epoch validation monitoring)
    if [[ "$LLM_NAME" == "qwen" && -n "$VALID_DATA_DIR" ]]; then
        TRAINING_ARGS+=("--validation_file" "${VALID_DATA_DIR}/valid.json")
    else
        TRAINING_ARGS+=("--validation_file" "${DATA_DIR}/valid.json")
    fi
    TRAINING_ARGS+=("--do_eval")
    if [[ "$MODEL_TYPE" == "t5" && "$LLM_NAME" != "gemini-2.5-flash-lite" && "$LLM_NAME" != "gemini-1.5-flash-8b" ]]; then
        TRAINING_ARGS+=("--val_column" "validation")
    fi
}

build_training_args

# Start training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=${GPU} python "${PYTHON_SCRIPT}" "${TRAINING_ARGS[@]}"

echo "Training completed for ${EPOCHS} epochs"

# Validation is now integrated into training (epoch-by-epoch validation monitoring)
echo "Validation was performed during training (epoch-by-epoch monitoring)"

# Run prediction for certain configurations
run_prediction() {
    if [[ ("$LLM_NAME" != "gemini-2.5-flash-lite" && "$LLM_NAME" != "gemini-1.5-flash-8b") || "$TYPE" != "dev" ]]; then
        echo "Starting prediction..."
        PREDICT_OUTPUT_DIR="${OUTPUT_DIR}/prediction"
        mkdir -p "${PREDICT_OUTPUT_DIR}"

        # Determine prediction file path
        if [[ "$LLM_NAME" == "qwen" ]]; then
            PREDICT_FILE="../data/full_silver_labeled/${ACTUAL_LLM_NAME}/predict.json"
        else
            PREDICT_FILE="../data/${DATASET_SIZE}/${ACTUAL_LLM_NAME}/predict.json"
        fi

        PREDICTION_ARGS=(
            "--model_name_or_path" "${OUTPUT_DIR}"
            "--validation_file" "${PREDICT_FILE}"
            "--question_column" "question"
            "--answer_column" "answer"
            "--max_seq_length" "384"
            "--per_device_eval_batch_size" "${EVAL_BATCH_SIZE}"
            "--output_dir" "${PREDICT_OUTPUT_DIR}"
            "--overwrite_cache"
            "--do_eval"
            "--seed" "42"
        )

        # Add model-specific arguments
        if [[ "$MODEL_TYPE" == "t5" ]]; then
            PREDICTION_ARGS+=("--source_prefix" "classify: ")
            if [[ "$LLM_NAME" != "gemini-2.5-flash-lite" && "$LLM_NAME" != "gemini-1.5-flash-8b" ]]; then
                PREDICTION_ARGS+=("--doc_stride" "128")
                PREDICTION_ARGS+=("--val_column" "validation")
            fi
        elif [[ "$MODEL_TYPE" == "bert" && ("$LLM_NAME" == "xl" || "$LLM_NAME" == "xxl") ]]; then
            PREDICTION_ARGS+=("--cache_dir" "../../.cache")
            PREDICTION_ARGS+=("--doc_stride" "128")
            PREDICTION_ARGS+=("--val_column" "validation")
        fi

        CUDA_VISIBLE_DEVICES=${GPU} python "${PYTHON_SCRIPT}" "${PREDICTION_ARGS[@]}"
        echo "Prediction completed for ${EPOCHS} epochs"
    fi
}

run_prediction

echo ""
echo "🎉 All tasks completed!"
echo "📁 Results saved in: ${OUTPUT_DIR}"
echo "📄 Check these files:"
echo "  - training_history.json: Complete training progression"
echo "  - validation/eval_results_epoch_X.json: Per-epoch validation results"
echo "  - logs.log: Detailed training logs with epoch summaries"
if [[ ("$LLM_NAME" != "gemini-2.5-flash-lite" && "$LLM_NAME" != "gemini-1.5-flash-8b") || "$TYPE" != "dev" ]]; then
    echo "📄 Check prediction results in: ${OUTPUT_DIR}/prediction/"
fi

