# scripts/bootstrap.sh
#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
VENV_DIR=".venv"
export PYTHONNOUSERSITE=1

echo ">>> Creating venv at ${VENV_DIR}"
[[ -d "${VENV_DIR}" ]] || ${PY} -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo ">>> Upgrading pip"
"${PY}" -m pip install --upgrade pip wheel setuptools

echo ">>> Installing requirements"
"${PY}" -m pip install -r requirements.txt

echo ">>> Ensuring project directories exist"
mkdir -p data experiments plots reports results scripts

echo ">>> Sanity checks"
${PY} - <<'PYCODE'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
print("✓ imports OK")
PYCODE

echo "✓ Done. Activate with:  source ${VENV_DIR}/bin/activate"
