#!/bin/bash
# Monitor BraTS training progress

echo "=== BraTS Training Monitor ==="
echo ""

# Check if training process is running
if ps aux | grep -q "[p]ython3 train.py"; then
    PID=$(ps aux | grep "[p]ython3 train.py" | awk '{print $2}')
    echo "✓ Training is RUNNING (PID: $PID)"
    echo ""

    # Show resource usage
    ps aux | grep "[p]ython3 train.py" | awk '{print "  CPU: "$3"% | Memory: "$4"% | Time: "$10}'
    echo ""
else
    echo "✗ Training is NOT running"
    echo ""
fi

# Show last 15 lines of log
echo "=== Latest Training Log ==="
tail -15 training.log
echo ""

# Count completed epochs
EPOCHS=$(grep -c "Epoch \[" training.log 2>/dev/null || echo "0")
echo "=== Progress ==="
echo "Completed epochs: $EPOCHS / 50"

# Show validation results if any
if grep -q "Val mean Dice" training.log 2>/dev/null; then
    echo ""
    echo "=== Latest Validation Results ==="
    grep "Val mean Dice" training.log | tail -3
fi

echo ""
echo "To view full log: tail -f training.log"
echo "To stop training: kill $PID"
