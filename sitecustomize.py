# sitecustomize.py
import os

# Force Transformers to PyTorch-only backends
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# Quieter logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF C++ logs

# Make OpenMP/thread noise calmer on mac
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Stop pytest from auto-loading 3rd-party plugins (dash/flask/etc.)
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
# Also ensure no residual plugin list is injected
os.environ.setdefault("PYTEST_PLUGINS", "")
