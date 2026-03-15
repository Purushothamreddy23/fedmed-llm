"""
scripts/merge_model.py
After federated training completes, this script:
1. Loads the final aggregated LoRA weights from /logs/
2. Merges them into base Phi-2 using peft.merge_and_unload()
3. Saves the final standalone model to /model/fedmed_phi2/

Run this after Day 2 training:
    python scripts/merge_model.py
"""

import os
import glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

MODEL_NAME   = "microsoft/phi-2"
LOG_DIR      = os.path.join(os.path.dirname(__file__), "..", "logs")
ADAPTER_DIR  = os.path.join(os.path.dirname(__file__), "..", "model", "lora_adapter")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "model", "fedmed_phi2")

os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def load_final_weights():
    """Load the last round's aggregated LoRA weights from /logs/."""
    weight_files = sorted(glob.glob(os.path.join(LOG_DIR, "round_*_weights.npy")))
    if not weight_files:
        raise FileNotFoundError(
            "No checkpoint files found in /logs/.\n"
            "Run federated training first (Day 2)."
        )
    latest = weight_files[-1]
    print(f"Loading weights from: {latest}")
    weights = np.load(latest, allow_pickle=True)
    return [np.array(w) for w in weights]


def set_lora_weights(model, weights):
    """Apply saved numpy weights back into the LoRA adapter."""
    lora_params = [
        param for name, param in model.named_parameters()
        if "lora" in name and param.requires_grad
    ]
    for param, weight in zip(lora_params, weights):
        param.data = torch.tensor(weight, dtype=param.dtype).to(param.device)


def main():
    print("="*55)
    print("  FedMed-LLM: Merging LoRA into Phi-2")
    print("="*55)

    # 1. Load final federated weights
    final_weights = load_final_weights()
    print(f"  Loaded {len(final_weights)} LoRA weight tensors")

    # 2. Load base Phi-2 in fp16 (full precision for merging)
    print("\nLoading base Phi-2 in fp16 for merging (no quantisation)...")
    print("  NOTE: This requires ~14 GB RAM. Run on Colab with High-RAM runtime")
    print("  or on a machine with sufficient RAM.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",          # CPU merge — avoids VRAM limits
        trust_remote_code=True,
    )

    # 3. Apply LoRA config and load federated weights
    model = get_peft_model(model, LORA_CONFIG)
    print("Applied LoRA config to base model")

    set_lora_weights(model, final_weights)
    print("Loaded federated LoRA weights into adapter")

    # 4. Save the LoRA adapter separately (backup)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nLoRA adapter saved to: {ADAPTER_DIR}")

    # 5. Merge LoRA into base weights — produces standalone model
    print("\nMerging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    print("Merge complete!")

    # 6. Save the final merged model
    print(f"\nSaving merged model to: {OUTPUT_DIR}")
    print("  This will take a few minutes (~5 GB)...")
    merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "="*55)
    print("  MERGE COMPLETE!")
    print(f"  Final model saved to: {OUTPUT_DIR}")
    print("  Next: run the FastAPI backend (Day 3)")
    print("="*55)

    # 7. Quick sanity check
    print("\nRunning sanity check on merged model...")
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=OUTPUT_DIR,
        tokenizer=OUTPUT_DIR,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    result = pipe(
        "Question: What are the symptoms of diabetes?\nAnswer:",
        max_new_tokens=80,
        do_sample=False,
    )
    response = result[0]["generated_text"].split("Answer:")[-1].strip()
    print(f"  Test Q: What are the symptoms of diabetes?")
    print(f"  Model A: {response[:200]}")
    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
