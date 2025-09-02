#!/bin/bash
# Smart Project Root Detection Script
# This script automatically detects the project root using multiple fallback strategies

detect_project_root() {
    local script_dir
    local current_dir
    local git_root
    local search_dir
    
    # Strategy 1: Use git repository root (most reliable)
    if command -v git >/dev/null 2>&1; then
        git_root=$(git rev-parse --show-toplevel 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$git_root" ]; then
            # If git root is Adaptive-RAG directory, project root is the parent
            if [ "$(basename "$git_root")" = "Adaptive-RAG" ]; then
                echo "$(dirname "$git_root")"
            else
                echo "$git_root"
            fi
            return 0
        fi
    fi
    
    # Strategy 2: Look for marker files starting from script location
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    search_dir="$script_dir"
    
    # Walk up the directory tree looking for marker files
    while [ "$search_dir" != "/" ]; do
        # Check for typical project root markers
        if [ -f "$search_dir/adaptive_rag_parallel.py" ] || \
           [ -f "$search_dir/run.py" ] || \
           [ -d "$search_dir/adaptive_rag_benchmark" ] || \
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
    
    # Strategy 3: Look for marker files starting from current directory
    current_dir="$(pwd)"
    search_dir="$current_dir"
    
    while [ "$search_dir" != "/" ]; do
        if [ -f "$search_dir/adaptive_rag_parallel.py" ] || \
           [ -f "$search_dir/run.py" ] || \
           [ -d "$search_dir/adaptive_rag_benchmark" ] || \
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
    
    # Strategy 4: Use environment variable if set
    if [ -n "$PROJECT_ROOT" ]; then
        echo "$PROJECT_ROOT"
        return 0
    fi
    
    # Strategy 5: Final fallback - use current directory
    echo "$(pwd)"
    return 1
}

# Function to set up PROJECT_ROOT automatically
setup_project_root() {
    if [ -z "$PROJECT_ROOT" ]; then
        export PROJECT_ROOT=$(detect_project_root)
        
        if [ $? -eq 0 ]; then
            echo "🔍 Auto-detected project root: $PROJECT_ROOT"
        else
            echo "⚠️  Could not reliably detect project root, using: $PROJECT_ROOT"
        fi
    else
        echo "📁 Using provided PROJECT_ROOT: $PROJECT_ROOT"
    fi
    
    # Validate that the detected path looks correct
    if [ ! -d "$PROJECT_ROOT/Adaptive-RAG" ]; then
        echo "❌ Warning: $PROJECT_ROOT doesn't contain Adaptive-RAG directory"
        echo "   You may need to set PROJECT_ROOT manually"
    fi
}

# Main execution if script is run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    setup_project_root
    echo "PROJECT_ROOT=$PROJECT_ROOT"
fi
