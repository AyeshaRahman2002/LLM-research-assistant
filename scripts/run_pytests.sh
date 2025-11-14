#!/usr/bin/env bash
set -euo pipefail

# Disable auto-loading of external pytest plugins (dash/flask/etc.)
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_PLUGINS=""

# Ensure project is on PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Run tests
python -m pytest -q "$@"
