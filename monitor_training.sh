#!/bin/bash
# Monitor training progress

echo "===================================="
echo "AINET Training Monitor"
echo "===================================="
echo ""

# Check training progress
echo "📊 Training Progress:"
/root/miniconda3/envs/ainet/bin/python check_training.py
echo ""

# Find latest checkpoint
LATEST=$(ls -td outputs/checkpoint-step-* 2>/dev/null | head -1)
if [ ! -z "$LATEST" ]; then
    echo "📁 Latest checkpoint: $(basename $LATEST)"
    echo ""
fi

# Check if training is running
TRAIN_PID=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$TRAIN_PID" ]; then
    echo "✅ Training is RUNNING (PID: $TRAIN_PID)"

    # Get training time
    START_TIME=$(ps -o lstart= -p $TRAIN_PID)
    echo "   Started: $START_TIME"
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "===================================="
echo "Quick Test Commands:"
echo "===================================="
echo ""
echo "# Test Text-to-Image:"
echo "/root/miniconda3/envs/ainet/bin/python inference.py \\"
echo "  --checkpoint $LATEST \\"
echo "  --mode t2i \\"
echo "  --text 'your description here' \\"
echo "  --output output.png \\"
echo "  --device cuda"
echo ""
echo "# Test Image-to-Text (when ready):"
echo "/root/miniconda3/envs/ainet/bin/python inference.py \\"
echo "  --checkpoint $LATEST \\"
echo "  --mode i2t \\"
echo "  --image path/to/image.jpg \\"
echo "  --device cuda"
echo ""
