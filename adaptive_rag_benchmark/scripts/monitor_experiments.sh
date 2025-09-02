#!/bin/bash

function kill_all_experiments() {
    echo "🛑 Killing all experiment processes..."
    
    # Kill any shell orchestrator processes
    ORCHESTRATOR_PIDS=$(ps aux | grep "run_experiments_nohup.sh" | grep -v grep | awk '{print $2}')
    if [ ! -z "$ORCHESTRATOR_PIDS" ]; then
        echo "🔪 Killing shell orchestrator processes: $ORCHESTRATOR_PIDS"
        kill $ORCHESTRATOR_PIDS 2>/dev/null
        sleep 2
        # Force kill if still running
        kill -9 $ORCHESTRATOR_PIDS 2>/dev/null
    fi
    
    # Kill any adaptive_rag_parallel processes
    WORKER_PIDS=$(ps aux | grep "adaptive_rag_parallel.py" | grep -v grep | awk '{print $2}')
    if [ ! -z "$WORKER_PIDS" ]; then
        echo "🔪 Killing worker processes: $WORKER_PIDS"
        kill $WORKER_PIDS 2>/dev/null
        sleep 2
        # Force kill if still running
        kill -9 $WORKER_PIDS 2>/dev/null
    fi
    
    # Kill any Python processes with 'comprehensive' or 'adaptive' in the command
    PYTHON_PIDS=$(ps aux | grep python | grep -E "(comprehensive|adaptive)" | grep -v grep | awk '{print $2}')
    if [ ! -z "$PYTHON_PIDS" ]; then
        echo "🔪 Killing related Python processes: $PYTHON_PIDS"
        kill $PYTHON_PIDS 2>/dev/null
        sleep 2
        kill -9 $PYTHON_PIDS 2>/dev/null
    fi
    
    # Clean up PID files
    if [ -f runs/experiments.pid ]; then
        echo "🧹 Removing PID file"
        rm -f runs/experiments.pid
    fi
    
    # Verify all processes are gone
    REMAINING=$(ps aux | grep -E "(run_comprehensive|adaptive_rag)" | grep -v grep | wc -l)
    if [ $REMAINING -eq 0 ]; then
        echo "✅ All experiment processes successfully terminated"
    else
        echo "⚠️  Warning: $REMAINING processes may still be running"
        ps aux | grep -E "(run_comprehensive|adaptive_rag)" | grep -v grep
    fi
    
    return 0
}

echo "🔍 Comprehensive Experiments Monitor"
echo "📅 Current time: $(date)"
echo ""

# Check for kill command
if [ "$1" == "kill" ] || [ "$1" == "--kill" ] || [ "$1" == "-k" ]; then
    kill_all_experiments
    exit 0
fi

# Check for help command
if [ "$1" == "help" ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "📋 Comprehensive Experiments Monitor - Help"
    echo ""
    echo "Usage:"
    echo "  ./monitor_experiments.sh           - Monitor experiment status"
    echo "  ./monitor_experiments.sh kill      - Kill all experiment processes"
    echo "  ./monitor_experiments.sh --help    - Show this help"
    echo ""
    echo "Kill options:"
    echo "  kill, --kill, -k                  - Terminate all processes"
    echo ""
    echo "The kill command will:"
    echo "  • Kill main orchestrator processes"
    echo "  • Kill all adaptive_rag_parallel worker processes"  
    echo "  • Kill any related Python processes"
    echo "  • Clean up PID files"
    echo "  • Verify all processes are terminated"
    echo ""
    echo "Files managed:"
    echo "  • experiments.pid - Main process ID tracking"
    echo "  • comprehensive_experiments_console.log - Console output"
    echo "  • comprehensive_experiments.log - Detailed logs"
    echo ""
    exit 0
fi

# Check if PID file exists
if [ ! -f runs/experiments.pid ]; then
    echo "❌ No runs/experiments.pid file found"
    echo "💡 Start experiments with: ./scripts/run_experiments_nohup.sh --background"
    exit 1
fi

PID=$(cat runs/experiments.pid)

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Experiments still running (PID: $PID)"
    
    # Show process info
    echo "📊 Process info:"
    ps -p $PID -o pid,ppid,cmd,etime,pcpu,pmem
    echo ""
    
    # Check progress from log files
    echo "📈 Recent progress:"
    if [ -f runs/sequential_experiments_console.log ]; then
        echo "--- Last 10 lines from console log ---"
        tail -10 runs/sequential_experiments_console.log
        echo ""
    elif [ -f runs/comprehensive_experiments_console.log ]; then
        echo "--- Last 10 lines from console log ---"
        tail -10 runs/comprehensive_experiments_console.log
        echo ""
    fi
    
    # Count completed experiments
    if [ -d runs/sequential_experiments_* ] || [ -d runs/comprehensive_experiments_* ]; then
        echo "📁 Output directories:"
        if [ -d runs/sequential_experiments_* ]; then
            ls -la runs/ | grep sequential_experiments_ | head -5
            EXP_DIR_PATTERN="runs/sequential_experiments_*"
        else
            ls -la runs/ | grep comprehensive_experiments_ | head -5
            EXP_DIR_PATTERN="runs/comprehensive_experiments_*"
        fi
        echo ""
        
        # Count completed experiments (individual experiment directories)
        COMPLETED_EXPERIMENTS=$(find $EXP_DIR_PATTERN -maxdepth 1 -type d -name "baseline_*" -o -name "adaptive_rag_*" 2>/dev/null | wc -l)
        echo "📊 Individual experiments completed so far: $COMPLETED_EXPERIMENTS"
        
        # Check for resource exhaustion logs
        RESOURCE_LOGS=$(find $EXP_DIR_PATTERN -name "resource_exhaustion.log" 2>/dev/null)
        if [ ! -z "$RESOURCE_LOGS" ]; then
            echo "📊 Resource exhaustion tracking available"
            for log in $RESOURCE_LOGS; do
                RESOURCE_ERRORS=$(grep -v "^#" "$log" 2>/dev/null | wc -l)
                if [ $RESOURCE_ERRORS -gt 0 ]; then
                    echo "  ⚠️  Resource errors found: $RESOURCE_ERRORS"
                    echo "  📝 Recent errors:"
                    tail -3 "$log" | sed 's/^/    /'
                else
                    echo "  ✅ No resource errors yet"
                fi
            done
        fi
        
        # Show disk usage
        echo "💾 Disk usage:"
        du -sh $EXP_DIR_PATTERN 2>/dev/null | head -5
    fi
    
    echo ""
    echo "🔄 To see live updates:"
    if [ -f runs/sequential_experiments_console.log ]; then
        echo "  tail -f runs/sequential_experiments_console.log"
    else
        echo "  tail -f runs/comprehensive_experiments_console.log"
    fi
    
else
    echo "❌ Experiments not running (PID: $PID was stopped)"
    
    # Check if there are any log files
    if [ -f runs/sequential_experiments_console.log ]; then
        echo "📝 Last few lines from console log:"
        tail -20 runs/sequential_experiments_console.log
        echo ""
    elif [ -f runs/comprehensive_experiments_console.log ]; then
        echo "📝 Last few lines from console log:"
        tail -20 runs/comprehensive_experiments_console.log
        echo ""
    fi
    
    # Check if completed
    if [ -d runs/sequential_experiments_* ] || [ -d runs/comprehensive_experiments_* ]; then
        echo "📊 Checking completion status..."
        
        if [ -d runs/sequential_experiments_* ]; then
            EXP_DIR_PATTERN="runs/sequential_experiments_*"
            COMPLETED_EXPERIMENTS=$(find $EXP_DIR_PATTERN -maxdepth 1 -type d -name "baseline_*" -o -name "adaptive_rag_*" 2>/dev/null | wc -l)
            echo "📈 Individual experiments completed: $COMPLETED_EXPERIMENTS / ~192"
            
            # Check for resource exhaustion summary
            RESOURCE_LOGS=$(find $EXP_DIR_PATTERN -name "resource_exhaustion.log" 2>/dev/null)
            if [ ! -z "$RESOURCE_LOGS" ]; then
                for log in $RESOURCE_LOGS; do
                    RESOURCE_ERRORS=$(grep -v "^#" "$log" 2>/dev/null | wc -l)
                    if [ $RESOURCE_ERRORS -gt 0 ]; then
                        echo "⚠️  Total resource errors: $RESOURCE_ERRORS"
                    else
                        echo "✅ No resource errors detected"
                    fi
                done
            fi
        else
            PROGRESS_FILES=$(find runs/comprehensive_experiments_* -name "*_metadata.json" 2>/dev/null | wc -l)
            echo "📈 Experiments completed: $PROGRESS_FILES / 192"
            
            if [ $PROGRESS_FILES -eq 192 ]; then
                echo "🎉 ALL EXPERIMENTS COMPLETED!"
            else
                echo "⚠️  Experiments stopped early - only $PROGRESS_FILES/192 completed"
            fi
        fi
        
        # Show summary files
        SUMMARY_FILES=$(find runs/*_experiments_* -name "final_summary.json" 2>/dev/null)
        if [ ! -z "$SUMMARY_FILES" ]; then
            echo "📊 Found summary files: $SUMMARY_FILES"
        fi
    fi
    
    echo ""
    echo "🔄 To restart experiments:"
    echo "  ./scripts/run_experiments_nohup.sh"
    echo ""
    echo "🛑 To force kill all experiments:"
    echo "  ./scripts/monitor_experiments.sh kill"
fi

echo ""
echo "💡 Available commands:"
echo "  ./scripts/monitor_experiments.sh          - Monitor status"
echo "  ./scripts/monitor_experiments.sh kill     - Kill all processes"
echo "  ./scripts/monitor_experiments.sh --help   - Show this help"
