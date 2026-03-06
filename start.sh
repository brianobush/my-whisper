#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env/bin/activate"
python "$SCRIPT_DIR/my-whisper.py"
