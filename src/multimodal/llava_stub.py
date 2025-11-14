# src/multimodal/llava_stub.py
"""
Stub interface for multimodal (image+text) input - mock of LLaVA.
"""
import json
from pathlib import Path


def main():
    demo = {
        "image": "example.jpg",
        "query": "Describe this image.",
        "response": "This is a stub multimodal model: describing a generic image.",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/llava_stub.json").write_text(json.dumps(demo, indent=2))
    print(json.dumps(demo, indent=2))


if __name__ == "__main__":
    main()
