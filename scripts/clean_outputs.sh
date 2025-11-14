# scripts/clean_outputs.sh
#!/usr/bin/env bash
set -euo pipefail
echo "==> Cleaning outputs (results, plots, reports, experiments artifacts)"
rm -rf results/* plots/* reports/* experiments/* data/uploads/*
mkdir -p results plots reports experiments data/uploads
echo "DONE Clean."
