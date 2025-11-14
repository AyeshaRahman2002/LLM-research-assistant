# src/llm_opt/triton_export.py
"""
Minimal TorchScript export for Triton-style serving (CPU-friendly).

- Wrap tiny CausalLM and return only logits (Tensor).
- Use torch.jit.trace on the wrapper to avoid HF scripting pitfalls.
- Saves BOTH:
    • results/triton/model.ts          (current path)
    • results/tiny_gpt2_traced.pt      (compat for tests)
"""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

DEFAULT_MODEL = "sshleifer/tiny-gpt2"  # tiny + CPU friendly


class LMLogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,  # outputs[0] = logits
        )
        return outputs[0]


def main():
    try:
        torch._C._jit_set_profiling_mode(False)
    except Exception:
        pass
    try:
        torch.jit.optimized_execution(False)
    except Exception:
        pass

    # Output paths
    out_dir = Path("results/triton")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ts_path = out_dir / "model.ts"
    compat_path = Path("results/tiny_gpt2_traced.pt")  # <- test expects this
    compat_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "export_meta.json"

    # Model/config
    config = AutoConfig.from_pretrained(DEFAULT_MODEL)
    config.use_cache = False
    config.return_dict = False
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, config=config).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapper = LMLogitsWrapper(model).eval()

    # Dummy inputs
    vocab = int(config.vocab_size)
    seq_len = 8
    batch = 1
    input_ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # Eager smoke
    with torch.no_grad():
        logits = wrapper(input_ids, attention_mask)
    assert logits.ndim == 3 and tuple(logits.shape) == (batch, seq_len, vocab)

    # Trace wrapper (returns a single Tensor)
    traced = torch.jit.trace(
        wrapper, (input_ids, attention_mask), check_trace=False, strict=False
    )

    # Save BOTH artifacts
    traced.save(str(out_ts_path))
    traced.save(str(compat_path))  # test looks for this

    # Load+run smoke on the .ts path
    ts = torch.jit.load(str(out_ts_path))
    with torch.no_grad():
        _ = ts(input_ids, attention_mask)

    # Metadata
    meta = {
        "model_name": DEFAULT_MODEL,
        "vocab_size": vocab,
        "seq_len": seq_len,
        "batch": batch,
        "artifact": str(out_ts_path),
        "compat_artifact": str(compat_path),
        "logits_shape": list(logits.shape),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"DONE TorchScript exported to: {out_ts_path}")
    print(f"DONE Compat artifact saved to: {compat_path}")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
