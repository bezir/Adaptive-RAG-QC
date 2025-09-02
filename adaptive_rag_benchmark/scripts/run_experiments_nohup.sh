#!/bin/bash

echo "Starting Sequential Adaptive RAG Experiments"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"

# Set up environment variables
# Automatically detect PROJECT_ROOT if not set
if [ -z "$PROJECT_ROOT" ]; then
    # Try to auto-detect from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
    export PROJECT_ROOT
    echo "Auto-detected PROJECT_ROOT: $PROJECT_ROOT"
fi
cd ${PROJECT_ROOT}/Adaptive-RAG

# Initialize conda properly for shell script
if [ -f "${PROJECT_ROOT}/miniconda3/etc/profile.d/conda.sh" ]; then
    source ${PROJECT_ROOT}/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
else
    echo "WARNING: Conda not found, trying alternative method..."
    source ~/.bashrc
    eval "$(conda shell.bash hook)" 2>/dev/null || echo "WARNING: Conda initialization warning (may still work)"
fi

conda activate adaptiverag
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set environment variables
# Set your API keys before running this script
# export GOOGLE_API_KEY="your_google_api_key_here"
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY environment variable is not set"
    echo "   Please set it before running: export GOOGLE_API_KEY=\"your_api_key\""
fi
export PYTHONUNBUFFERED=1  # For immediate output in nohup

# Fixed configuration - Optimized for 40 Qwen servers
WORKERS=40
MAX_QUERIES=500
SEED=42

# Experiment configuration
GENERATORS=("gemini-2.5-flash-lite" "gemini-1.5-flash-8b" "Qwen/Qwen2.5-3B-Instruct")
DATASETS=("musique" "2wikimultihopqa" "trivia" "squad" "hotpotqa" "nq")
BASELINE_STRATEGIES=("nor" "oner" "ircot")

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="adaptive_rag_benchmark/runs/sequential_experiments_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Environment setup complete"
echo "Configuration:"
echo "  Workers: ${WORKERS} (40 samples processed in parallel per experiment - all Qwen servers)"
echo "  Max queries: ${MAX_QUERIES}"
echo "  Seed: ${SEED}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Create resource exhaustion tracking file
RESOURCE_LOG="${OUTPUT_DIR}/resource_exhaustion.log"
echo "# Resource Exhaustion Tracking" > "$RESOURCE_LOG"
echo "# Timestamp | Experiment | Error Count | Error Details" >> "$RESOURCE_LOG"

# Function to find all classifier paths with their types
find_classifiers_with_types() {
    local generator=$1
    local classifiers=()
    
    # BERT classifiers
    if [ -d "classifier/outputs/bert-large" ]; then
        for dir in classifier/outputs/bert-large/*/; do
            if [ -d "$dir" ]; then
                dirname=$(basename "$dir")
                if [[ "$dirname" == *"$generator"* ]]; then
                    classifiers+=("bert:$dir")
                elif [[ "$generator" == "gemini-1.5-flash-8b" && "$dirname" == *"gemini-1.5-8b"* ]]; then
                    classifiers+=("bert:$dir")
                elif [[ "$generator" == "Qwen/Qwen2.5-3B-Instruct" && "$dirname" == *"qwen-2.5-3b"* ]]; then
                    classifiers+=("bert:$dir")
                fi
            fi
        done
    fi
    
    # T5 classifiers  
    if [ -d "classifier/outputs/t5-large" ]; then
        for dir in classifier/outputs/t5-large/*/; do
            if [ -d "$dir" ]; then
                dirname=$(basename "$dir")
                if [[ "$dirname" == *"$generator"* ]]; then
                    classifiers+=("flan:$dir")
                elif [[ "$generator" == "gemini-1.5-flash-8b" && "$dirname" == *"gemini-1.5-8b"* ]]; then
                    classifiers+=("flan:$dir")
si                elif [[ "$generator" == "Qwen/Qwen2.5-3B-Instruct" && "$dirname" == *"qwen-2.5-3b"* ]]; then
                    classifiers+=("flan:$dir")
                fi
            fi
        done
    fi
    
    echo "${classifiers[@]}"
}

# Function to run a single experiment
run_experiment() {
    local exp_id=$1
    local exp_type=$2
    local generator=$3
    local dataset=$4
    local classifier_path=$5
    local classification_llm=$6
    local force_strategy=$7
    local verify_classification=${8:-"false"}
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Running Experiment: ${exp_id}"
    echo "📋 Type: ${exp_type}"
    echo "🤖 Generator: ${generator}"
    echo "📊 Dataset: ${dataset}"
    if [ -n "$classifier_path" ]; then
        if [[ "$exp_type" == *"bert"* ]]; then
            echo "🧠 Classifier: BERT - $(basename "$classifier_path")"
        elif [[ "$exp_type" == *"flan"* ]]; then
            echo "🧠 Classifier: FLAN-T5 - $(basename "$classifier_path")"
        else
            echo "🧠 Classifier: $(basename "$classifier_path")"
        fi
    elif [ "$classification_llm" = "true" ]; then
        echo "Classifier: LLM (self-classification)"
    elif [ -n "$force_strategy" ]; then
        echo "Strategy: ${force_strategy}"
    fi
    if [ "$verify_classification" = "true" ]; then
        echo "Enhanced Verification: ENABLED (ML Context + Probability Analysis)"
    fi
    echo "⚡ Workers: ${WORKERS}"
    echo "📝 Queries: ${MAX_QUERIES}"
    echo "🕐 Started: $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Create experiment directory
    local exp_dir="${OUTPUT_DIR}/${exp_id}"
    mkdir -p "$exp_dir"
    
    # Build command
    local cmd=(
        "python" "adaptive_rag_benchmark/adaptive_rag_parallel.py"
        "--model" "$generator"
        "--dataset" "$dataset"
        "--max_queries" "$MAX_QUERIES"
        "--workers" "$WORKERS"
        "--seed" "$SEED"
        "--output_dir" "$exp_dir"
        "--experiment_name" "$exp_id"
        "--verbose"
    )
    
    # Add port range for Qwen models - All 40 ports
    if [[ "$generator" == *"Qwen"* ]]; then
        cmd+=("--port-range" "8010-8049")
    fi
    
    # Add experiment-specific arguments
    if [ -n "$classifier_path" ]; then
        cmd+=("--classifier_path" "$classifier_path")
    elif [ "$classification_llm" = "true" ]; then
        cmd+=("--classification_llm")
    elif [ -n "$force_strategy" ]; then
        cmd+=("--force" "$force_strategy")
    fi
    
    # Add verification flag if enabled
    if [ "$verify_classification" = "true" ]; then
        cmd+=("--verify_classification")
    fi
    
    # Create log file
    local log_file="${exp_dir}/${exp_id}.log"
    
    # Run experiment and capture start time
    local start_time=$(date +%s)
    echo "Command: ${cmd[*]}" > "$log_file"
    echo "Started: $(date)" >> "$log_file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$log_file"
    
    # Execute the experiment
    "${cmd[@]}" >> "$log_file" 2>&1
    local exit_code=$?
    
    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    # Check for resource exhaustion errors in the log - fix regex to avoid false matches
    local resource_errors=0
    if [ -f "$log_file" ]; then
        # Use more specific patterns to avoid matching debug lines with "limit" in question IDs
        resource_errors=$(grep -c -E "(resource.{1,10}exhausted|rate.{1,5}limit|API.{1,5}limit|HTTP 429|quota.{1,10}exceeded|ResourceExhausted|RESOURCE_EXHAUSTED)" "$log_file" 2>/dev/null || echo 0)
    fi
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: ${exp_id} in ${elapsed_min}m ${elapsed_sec}s"
        if [ $resource_errors -gt 0 ]; then
            echo "⚠️  Resource exhaustion errors: ${resource_errors}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') | ${exp_id} | ${resource_errors} | Completed with errors" >> "$RESOURCE_LOG"
        fi
    else
        echo "❌ Failed: ${exp_id} after ${elapsed_min}m ${elapsed_sec}s (exit code: ${exit_code})"
        if [ $resource_errors -gt 0 ]; then
            echo "🚨 Resource exhaustion errors: ${resource_errors}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') | ${exp_id} | ${resource_errors} | Failed with resource errors" >> "$RESOURCE_LOG"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') | ${exp_id} | 0 | Failed with other errors" >> "$RESOURCE_LOG"
        fi
    fi
    
    # Add separator
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    return $exit_code
}

# Function to run all experiments sequentially
run_all_experiments() {
    local total_experiments=0
    local successful=0
    local failed=0
    local start_time=$(date +%s)
    
    echo "📊 Generating experiment list..."
    
    # Count total experiments first
    for generator in "${GENERATORS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            # Baseline experiments
            for strategy in "${BASELINE_STRATEGIES[@]}"; do
                ((total_experiments++))
            done
            
            # Adaptive RAG experiments (BERT and FLAN classifiers)
            local classifiers_with_types=($(find_classifiers_with_types "$generator"))
            for classifier_entry in "${classifiers_with_types[@]}"; do
                ((total_experiments++))
                # Add verification experiments for each classifier
                ((total_experiments++))
            done
            
            # Adaptive LLM experiments
            ((total_experiments++))
        done
    done
    
    echo "Total experiments to run: ${total_experiments}"
    echo "Estimated time: $((total_experiments * 10 / 60)) hours (10 min avg per experiment with 40 workers + enhanced retry)"
    echo "Enhanced retry system: Multi-port recovery, circuit breakers, load balancing"
    echo "Optimized: 40 parallel workers using all Qwen servers for maximum throughput"
    echo "Debug mode: Extensive logging enabled to track retry behavior"
    echo "Enhanced Verification: ML Context + Probability Analysis included"
    echo ""
    
    local current=0


    for generator in "${GENERATORS[@]}"; do
        local classifiers_with_types=($(find_classifiers_with_types "$generator"))
        echo "📚 Found ${#classifiers_with_types[@]} classifiers for ${generator}"
        
        for classifier_entry in "${classifiers_with_types[@]}"; do
            # Parse classifier type and path
            local classifier_type="${classifier_entry%%:*}"
            local classifier_path="${classifier_entry#*:}"
            local classifier_name=$(basename "$classifier_path")
            
            # ONLY run FLAN-T5 experiments in this section
            if [ "$classifier_type" == "flan" ]; then
                for dataset in "${DATASETS[@]}"; do
                    ((current++))
                    local exp_id="adaptive_rag_${classifier_type}_${generator//-/_}_${classifier_name}_${dataset}"
                    
                    echo "📈 Progress: ${current}/${total_experiments}"
                    echo "🔍 FLAN-T5 EXPERIMENT: ${exp_id}"
                    
                    if run_experiment "$exp_id" "adaptive_rag_${classifier_type}" "$generator" "$dataset" "$classifier_path" "false" ""; then
                        ((successful++))
                    else
                        ((failed++))
                    fi
                    
                    # Show progress  
                    local elapsed=$(($(date +%s) - start_time))
                    local avg_time=$((elapsed / current))
                    local remaining=$(((total_experiments - current) * avg_time))
                    echo "⏱️  Progress: ${current}/${total_experiments} | Success: ${successful} | Failed: ${failed} | ETA: $((remaining / 3600))h $((remaining % 3600 / 60))m"
                done
            fi
        done
    done
    
    # Run baseline experiments
    echo "🎯 Starting Baseline Experiments..."
    for generator in "${GENERATORS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for strategy in "${BASELINE_STRATEGIES[@]}"; do
                ((current++))
                local exp_id="baseline_${generator//-/_}_${dataset}_${strategy}"
                
                echo "📈 Progress: ${current}/${total_experiments}"
                
                if run_experiment "$exp_id" "baseline" "$generator" "$dataset" "" "false" "$strategy" "false"; then
                    ((successful++))
                else
                    ((failed++))
                fi
                
                # Show progress
                local elapsed=$(($(date +%s) - start_time))
                local avg_time=$((elapsed / current))
                local remaining=$(((total_experiments - current) * avg_time))
                echo "⏱️  Progress: ${current}/${total_experiments} | Success: ${successful} | Failed: ${failed} | ETA: $((remaining / 3600))h $((remaining % 3600 / 60))m"
            done
        done
    done
    
    # Run remaining adaptive RAG experiments (BERT classifiers only)
    echo "🧠 Starting BERT Classifier Experiments..."
    for generator in "${GENERATORS[@]}"; do
        local classifiers_with_types=($(find_classifiers_with_types "$generator"))
        echo "📚 Found ${#classifiers_with_types[@]} classifiers for ${generator}"
        
        for classifier_entry in "${classifiers_with_types[@]}"; do
            # Parse classifier type and path
            local classifier_type="${classifier_entry%%:*}"
            local classifier_path="${classifier_entry#*:}"
            local classifier_name=$(basename "$classifier_path")
            
            # ONLY run BERT experiments in this section (FLAN already done above)
            if [ "$classifier_type" == "bert" ]; then
                for dataset in "${DATASETS[@]}"; do
                    ((current++))
                    local exp_id="adaptive_rag_${classifier_type}_${generator//-/_}_${classifier_name}_${dataset}"
                    
                    echo "Progress: ${current}/${total_experiments}"
                    
                    if run_experiment "$exp_id" "adaptive_rag_${classifier_type}" "$generator" "$dataset" "$classifier_path" "false" "" "false"; then
                        ((successful++))
                    else
                        ((failed++))
                    fi
                    
                    # Run verification experiment
                    ((current++))
                    local verification_exp_id="verification_${classifier_type}_${generator//-/_}_${classifier_name}_${dataset}"
                    echo "Progress: ${current}/${total_experiments}"
                    echo "Enhanced Verification BERT EXPERIMENT: ${verification_exp_id}"
                    
                    if run_experiment "$verification_exp_id" "verification_${classifier_type}" "$generator" "$dataset" "$classifier_path" "false" "" "true"; then
                        ((successful++))
                    else
                        ((failed++))
                    fi
                    
                    # Show progress  
                    local elapsed=$(($(date +%s) - start_time))
                    local avg_time=$((elapsed / current))
                    local remaining=$(((total_experiments - current) * avg_time))
                    echo "⏱️  Progress: ${current}/${total_experiments} | Success: ${successful} | Failed: ${failed} | ETA: $((remaining / 3600))h $((remaining % 3600 / 60))m"
                done
            fi
        done
    done
    
    # Run adaptive LLM experiments
    echo "🤖 Starting Adaptive RAG LLM Experiments..."
    for generator in "${GENERATORS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            ((current++))
            local exp_id="adaptive_rag_llm_${generator//-/_}_${dataset}"
            
            echo "📈 Progress: ${current}/${total_experiments}"
            
            if run_experiment "$exp_id" "adaptive_rag_llm" "$generator" "$dataset" "" "true" "" "false"; then
                ((successful++))
            else
                ((failed++))
            fi
            
            # Show progress
            local elapsed=$(($(date +%s) - start_time))
            local avg_time=$((elapsed / current))
            local remaining=$(((total_experiments - current) * avg_time))
            echo "⏱️  Progress: ${current}/${total_experiments} | Success: ${successful} | Failed: ${failed} | ETA: $((remaining / 3600))h $((remaining % 3600 / 60))m"
        done
    done
    
    # Final summary
    local total_time=$(($(date +%s) - start_time))
    local total_hours=$((total_time / 3600))
    local total_min=$((total_time % 3600 / 60))
    
    echo ""
    echo "🎉 ALL EXPERIMENTS COMPLETE!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Final Results:"
    echo "  Total experiments: ${total_experiments}"
    echo "  Successful: ${successful}"
    echo "  Failed: ${failed}"
    echo "  Total time: ${total_hours}h ${total_min}m"
    echo "  Average per experiment: $((total_time / total_experiments / 60))m"
    echo ""
    echo "📂 Results saved to: ${OUTPUT_DIR}"
    echo "🚨 Resource exhaustion log: ${RESOURCE_LOG}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Check if running as background job
if [ "$1" = "--background" ]; then
    # Run in background with nohup
    echo "🔄 Running experiments in background..."
    
    # Save PID for monitoring
    {
        run_all_experiments
    } > "adaptive_rag_benchmark/runs/sequential_experiments_console.log" 2>&1 &
    
    PID=$!
    echo $PID > adaptive_rag_benchmark/runs/experiments.pid
    echo "💾 PID saved to adaptive_rag_benchmark/runs/experiments.pid: $PID"
    echo "📝 Console output: adaptive_rag_benchmark/runs/sequential_experiments_console.log"
echo ""
echo "🔍 To monitor progress:"
    echo "  tail -f adaptive_rag_benchmark/runs/sequential_experiments_console.log"
    echo "  cd adaptive_rag_benchmark && ./scripts/monitor_experiments.sh"
echo ""
echo "🛑 To stop experiments:"
    echo "  cd adaptive_rag_benchmark && ./scripts/monitor_experiments.sh kill"

# Wait a moment and check if it started properly
    sleep 3
if ps -p $PID > /dev/null; then
        echo "✅ Sequential experiments running successfully (PID: $PID)"
    else
        echo "❌ Experiments failed to start - check runs/sequential_experiments_console.log"
        exit 1
    fi
else
    # Run directly (foreground)
    echo "🔄 Running experiments in foreground..."
    echo "💡 To run in background, use: $0 --background"
    echo ""
    run_all_experiments
fi
