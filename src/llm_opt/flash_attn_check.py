# src/llm_opt/flash_attn_check.py
"""
Check whether FlashAttention is available & benchmark minimal matmul path.
Writes -> results/flash_attn_check.json
"""
import json
import time
from pathlib import Path

import torch


def main():
    avail = torch.cuda.is_available() and hasattr(
        torch.nn.functional, "scaled_dot_product_attention"
    )
    N = 256
    q = torch.randn(1, 8, N, 64)
    k = torch.randn(1, 8, N, 64)
    v = torch.randn(1, 8, N, 64)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()

    Path("results").mkdir(exist_ok=True)
    out = {"flash_attention_available": avail, "elapsed_sec": round(t1 - t0, 4)}
    Path("results/flash_attn_check.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
