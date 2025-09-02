#!/bin/bash
# Professional Environment Setup Script for Adaptive RAG Project
# This script provides secure and configurable environment setup

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for professional output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Function to log messages professionally
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to detect project root automatically
detect_project_root() {
    local script_dir
    local search_dir
    
    # Get script directory
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    search_dir="$script_dir"
    
    # Walk up the directory tree looking for project markers
    while [ "$search_dir" != "/" ]; do
        if [ -d "$search_dir/adaptive_rag_benchmark" ] || \
           [ -d "$search_dir/classifier" ] || \
           [ -d "$search_dir/scaled_silver_labeling" ]; then
            # If we found Adaptive-RAG directory, the project root is the parent
            if [ "$(basename "$search_dir")" = "Adaptive-RAG" ]; then
                echo "$(dirname "$search_dir")"
            else
                echo "$search_dir"
            fi
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done
    
    return 1
}

# Function to setup PROJECT_ROOT
setup_project_root() {
    if [ -z "${PROJECT_ROOT:-}" ]; then
        if PROJECT_ROOT=$(detect_project_root); then
            export PROJECT_ROOT
            log_info "Auto-detected PROJECT_ROOT: $PROJECT_ROOT"
        else
            log_error "Could not detect PROJECT_ROOT automatically"
            log_error "Please set PROJECT_ROOT environment variable manually:"
            log_error "  export PROJECT_ROOT=/path/to/your/project/root"
            return 1
        fi
    else
        log_info "Using provided PROJECT_ROOT: $PROJECT_ROOT"
    fi
    
    # Validate PROJECT_ROOT
    if [ ! -d "$PROJECT_ROOT/Adaptive-RAG" ]; then
        log_error "PROJECT_ROOT validation failed: $PROJECT_ROOT/Adaptive-RAG does not exist"
        return 1
    fi
    
    return 0
}

# Function to setup derived paths
setup_derived_paths() {
    export ADAPTIVE_RAG_ROOT="${PROJECT_ROOT}/Adaptive-RAG"
    export CONDA_ROOT="${PROJECT_ROOT}/miniconda3"
    export CACHE_DIR="${ADAPTIVE_RAG_ROOT}/.cache"
    export HF_HOME="${CACHE_DIR}/huggingface"
    export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface"
    export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    
    # Create cache directories
    mkdir -p "$CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
}

# Function to check API keys
check_api_keys() {
    local missing_keys=()
    
    if [ -z "${GOOGLE_API_KEY:-}" ]; then
        missing_keys+=("GOOGLE_API_KEY")
    fi
    
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        missing_keys+=("OPENAI_API_KEY")
    fi
    
    if [ ${#missing_keys[@]} -gt 0 ]; then
        log_warn "Missing API keys: ${missing_keys[*]}"
        log_warn "Set them before running experiments:"
        for key in "${missing_keys[@]}"; do
            log_warn "  export $key=\"your_${key,,}_here\""
        done
        return 1
    fi
    
    log_success "All required API keys are set"
    return 0
}

# Function to validate conda environment
validate_conda_environment() {
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        log_warn "No conda environment detected"
        log_warn "Recommended: conda activate adaptiverag"
        return 1
    fi
    
    if [ "$CONDA_DEFAULT_ENV" != "adaptiverag" ]; then
        log_warn "Current conda environment: $CONDA_DEFAULT_ENV"
        log_warn "Recommended environment: adaptiverag"
        return 1
    fi
    
    log_success "Conda environment validated: $CONDA_DEFAULT_ENV"
    return 0
}

# Main setup function
main() {
    log_info "Starting environment setup for Adaptive RAG project"
    
    # Setup PROJECT_ROOT
    if ! setup_project_root; then
        return 1
    fi
    
    # Setup derived paths
    setup_derived_paths
    
    # Display configuration
    log_success "Environment configured successfully:"
    echo "  PROJECT_ROOT: $PROJECT_ROOT"
    echo "  ADAPTIVE_RAG_ROOT: $ADAPTIVE_RAG_ROOT"
    echo "  CACHE_DIR: $CACHE_DIR"
    
    # Optional validations (non-blocking)
    check_api_keys || true
    validate_conda_environment || true
    
    log_success "Environment setup complete"
}

# Execute main function if script is run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi



