#!/usr/bin/env bash
# Quickstart script to launch the CCTV AI Head Counter CLI
cd "$(dirname "$0")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 cli.py "$@"

