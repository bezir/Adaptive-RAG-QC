#!/bin/bash

# Master Labeling Script
# Unified script to run labeling experiments with flexible parameters
#
# Usage: ./run_master_labeling.sh --model MODEL --strategy STRATEGY --size SIZE --dataset DATASET [--debug]
#
# Parameters:
#   --model: gemini-2.5-flash-lite | gemini-1.5-flash-8b
#   --strategy: optimized | original
#   --size: any positive integer
#   --dataset: all | hotpotqa | musique | 2wikimultihopqa | squad | nq | trivia
#   --debug: enable detailed logging (optional)

set -e  # Exit on any error

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
MODEL=""
STRATEGY=""
SIZE=""
DATASET=""
WORKERS=8
DEBUG=false

# Available options
AVAILABLE_MODELS=("gemini-2.5-flash-lite" "gemini-1.5-flash-8b")
AVAILABLE_STRATEGIES=("optimized" "original")
AVAILABLE_DATASETS=("hotpotqa" "musique" "2wikimultihopqa" "squad" "nq" "trivia")

# Function to print usage
print_usage() {
    echo -e "${CYAN}Master Labeling Script${NC}"
    echo -e "${YELLOW}Usage:${NC} $0 --model MODEL --strategy STRATEGY --size SIZE --dataset DATASET [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Required Parameters:${NC}"
    echo -e "  ${BLUE}--model${NC}     Model to use:"
    echo "              • gemini-2.5-flash-lite (default, faster)"
    echo "              • gemini-1.5-flash-8b (alternative)"
    echo -e "  ${BLUE}--strategy${NC}  Labeling strategy:"
    echo "              • optimized (sequential pipeline: NOR → ONER, faster)"
    echo "              • original (full system: NOR + ONER + IRCOT, comprehensive)"
    echo -e "  ${BLUE}--size${NC}      Sample size (any positive integer)"
    echo -e "  ${BLUE}--dataset${NC}   Dataset(s) to process:"
    echo "              • all (runs all 6 datasets)"
    echo "              • hotpotqa, musique, 2wikimultihopqa, squad, nq, trivia"
    echo ""
    echo -e "${YELLOW}Optional Parameters:${NC}"
    echo -e "  ${BLUE}--workers${NC}   Number of parallel workers (default: 8)"
    echo -e "  ${BLUE}--debug${NC}     Enable detailed logging (shows API calls and processing info)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Run optimized labeling on all datasets with 1000 samples"
    echo "  $0 --model gemini-2.5-flash-lite --strategy optimized --size 1000 --dataset all"
    echo ""
    echo "  # Run original labeling on HotpotQA with 500 samples"
    echo "  $0 --model gemini-1.5-flash-8b --strategy original --size 500 --dataset hotpotqa"
    echo ""
    echo "  # Run with debug logging enabled"
    echo "  $0 --model gemini-2.5-flash-lite --strategy optimized --size 100 --dataset hotpotqa --debug"
}

# Function to validate parameters
validate_parameters() {
    local errors=()
    
    # Check if all required parameters are provided
    if [[ -z "$MODEL" ]]; then
        errors+=("Missing required parameter: --model")
    fi
    if [[ -z "$STRATEGY" ]]; then
        errors+=("Missing required parameter: --strategy")
    fi
    if [[ -z "$SIZE" ]]; then
        errors+=("Missing required parameter: --size")
    fi
    if [[ -z "$DATASET" ]]; then
        errors+=("Missing required parameter: --dataset")
    fi
    
    # Validate model
    if [[ -n "$MODEL" ]]; then
        local valid_model=false
        for valid in "${AVAILABLE_MODELS[@]}"; do
            if [[ "$MODEL" == "$valid" ]]; then
                valid_model=true
                break
            fi
        done
        if [[ "$valid_model" == false ]]; then
            errors+=("Invalid model '$MODEL'. Available models: ${AVAILABLE_MODELS[*]}")
        fi
    fi
    
    # Validate strategy
    if [[ -n "$STRATEGY" ]]; then
        local valid_strategy=false
        for valid in "${AVAILABLE_STRATEGIES[@]}"; do
            if [[ "$STRATEGY" == "$valid" ]]; then
                valid_strategy=true
                break
            fi
        done
        if [[ "$valid_strategy" == false ]]; then
            errors+=("Invalid strategy '$STRATEGY'. Available strategies: ${AVAILABLE_STRATEGIES[*]}")
        fi
    fi
    
    # Validate size
    if [[ -n "$SIZE" ]]; then
        if ! [[ "$SIZE" =~ ^[1-9][0-9]*$ ]]; then
            errors+=("Invalid size '$SIZE'. Must be a positive integer.")
        fi
    fi
    
    # Validate dataset
    if [[ -n "$DATASET" && "$DATASET" != "all" ]]; then
        local valid_dataset=false
        for valid in "${AVAILABLE_DATASETS[@]}"; do
            if [[ "$DATASET" == "$valid" ]]; then
                valid_dataset=true
                break
            fi
        done
        if [[ "$valid_dataset" == false ]]; then
            errors+=("Invalid dataset '$DATASET'. Available datasets: all, ${AVAILABLE_DATASETS[*]}")
        fi
    fi
    
    # Validate workers
    if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
        errors+=("Invalid workers count '$WORKERS'. Must be a positive integer.")
    fi
    
    # Print errors if any
    if [[ ${#errors[@]} -gt 0 ]]; then
        echo -e "${RED}❌ Validation errors:${NC}"
        for error in "${errors[@]}"; do
            echo -e "   ${RED}•${NC} $error"
        done
        echo ""
        print_usage
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    echo -e "${BLUE}🔧 Setting up environment...${NC}"
    
    # Ensure we're using bash
    if [ -z "$BASH_VERSION" ]; then
        echo -e "${RED}❌ This script requires bash. Please run with: bash $0${NC}"
        exit 1
    fi

    # Initialize conda if not already done
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ Conda not found. Please ensure conda is installed and in PATH.${NC}"
        exit 1
    fi

    # Try to activate conda environment
    if conda info --envs | grep -q adaptiverag; then
        echo -e "${GREEN}   Activating conda environment: adaptiverag${NC}"
        eval "$(conda shell.bash hook)"
        conda activate adaptiverag
    else
        echo -e "${RED}❌ Conda environment 'adaptiverag' not found. Please create it first.${NC}"
        exit 1
    fi

    # Change to project directory
    # Set up environment variables
    export PROJECT_ROOT=${PROJECT_ROOT:-""}
    cd ${PROJECT_ROOT}/Adaptive-RAG || {
        echo -e "${RED}❌ Failed to change to project directory${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Environment setup complete${NC}"
}

# Function to run labeling for a single dataset
run_single_dataset() {
    local dataset=$1
    local dataset_start dataset_end dataset_duration
    
    echo ""
    echo -e "${CYAN}📊 Processing dataset: ${YELLOW}$dataset${NC}"
    echo -e "${BLUE}⏰ Started at: $(date)${NC}"
    
    dataset_start=$(date +%s)
    
    # Build command with optional debug flag
    local cmd="python scaled_silver_labeling/scripts/run_unified_labeling.py \
        --model \"$MODEL\" \
        --dataset \"$dataset\" \
        --strategy \"$STRATEGY\" \
        --sample_size \"$SIZE\" \
        --workers \"$WORKERS\""
    
    # Add debug flag if enabled
    if [[ "$DEBUG" == true ]]; then
        cmd="$cmd --debug"
    fi
    
    # Execute the command
    eval "$cmd"
    
    local exit_code=$?
    dataset_end=$(date +%s)
    dataset_duration=$((dataset_end - dataset_start))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $dataset completed successfully in ${dataset_duration}s${NC}"
    else
        echo -e "${RED}❌ $dataset failed with exit code $exit_code${NC}"
        return $exit_code
    fi
    
    echo -e "${BLUE}⏰ Finished at: $(date)${NC}"
    echo -e "${YELLOW}--------------------------------------------------${NC}"
    
    return 0
}

# Function to run labeling process
run_labeling() {
    local start_time end_time total_duration total_minutes
    local datasets_to_process=()
    local failed_datasets=()
    local successful_datasets=()
    
    # Determine which datasets to process
    if [[ "$DATASET" == "all" ]]; then
        datasets_to_process=("${AVAILABLE_DATASETS[@]}")
        echo -e "${CYAN}🚀 Starting unified labeling for ALL 6 datasets${NC}"
    else
        datasets_to_process=("$DATASET")
        echo -e "${CYAN}🚀 Starting unified labeling for dataset: ${YELLOW}$DATASET${NC}"
    fi
    
    echo -e "${YELLOW}Configuration:${NC}"
    echo -e "   Model: ${GREEN}$MODEL${NC}"
    echo -e "   Strategy: ${GREEN}$STRATEGY${NC}"
    echo -e "   Sample size: ${GREEN}$SIZE${NC}"
    echo -e "   Workers: ${GREEN}$WORKERS${NC}"
    echo -e "   Debug mode: ${GREEN}$(if [[ "$DEBUG" == true ]]; then echo "enabled"; else echo "disabled"; fi)${NC}"
    echo -e "   Datasets: ${GREEN}${datasets_to_process[*]}${NC}"
    echo -e "${YELLOW}==================================================${NC}"
    
    # Track timing
    start_time=$(date +%s)
    
    # Process each dataset
    for dataset in "${datasets_to_process[@]}"; do
        if run_single_dataset "$dataset"; then
            successful_datasets+=("$dataset")
        else
            failed_datasets+=("$dataset")
        fi
    done
    
    # Calculate total time
    end_time=$(date +%s)
    total_duration=$((end_time - start_time))
    total_minutes=$((total_duration / 60))
    
    # Print summary
    echo ""
    echo -e "${CYAN}🎉 Labeling process completed!${NC}"
    echo -e "${BLUE}⏱️  Total time: ${total_duration}s (${total_minutes} minutes)${NC}"
    echo -e "${BLUE}📁 Results saved in scaled_silver_labeling/predictions/${NC}"
    echo ""
    
    if [[ ${#successful_datasets[@]} -gt 0 ]]; then
        echo -e "${GREEN}✅ Successful datasets (${#successful_datasets[@]}):${NC}"
        for dataset in "${successful_datasets[@]}"; do
            echo -e "   ${GREEN}•${NC} $dataset"
        done
    fi
    
    if [[ ${#failed_datasets[@]} -gt 0 ]]; then
        echo -e "${RED}❌ Failed datasets (${#failed_datasets[@]}):${NC}"
        for dataset in "${failed_datasets[@]}"; do
            echo -e "   ${RED}•${NC} $dataset"
        done
        echo ""
        echo -e "${YELLOW}⚠️  Some datasets failed. Check logs for details.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}==================================================${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown parameter: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${CYAN}🚀 Master Labeling Script${NC}"
echo -e "${YELLOW}==================================================${NC}"

# Validate parameters
validate_parameters

# Setup environment
setup_environment

# Run labeling process
run_labeling

echo -e "${GREEN}🎉 All done!${NC}"
