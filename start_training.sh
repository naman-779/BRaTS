#!/bin/bash
cd "/Users/namantaneja/Library/Mobile Documents/com~apple~CloudDocs/Desktop/My Mac/Work/College/Research/implementBRaTS"
source venv/bin/activate
python3 train.py --config config.yaml > training.log 2>&1 &
echo "Training started with PID: $!"
