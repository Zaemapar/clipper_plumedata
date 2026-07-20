#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
echo "Clearing old environments..."
rm -rf .venv
echo "Setting up virtual environment in macOS..."
python3 -m venv .venv
echo "Installing required packages..."
.venv/bin/pip install -r src/requirements.txt
echo "Generating launcher..."
cat << 'EOF' > ecspa.command
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Launching ECSPA..."
cd "$DIR/src"
../.venv/bin/python3 analysis_main.py
EOF
echo "Setup complete. Double-click ecspa.command to run the main application."