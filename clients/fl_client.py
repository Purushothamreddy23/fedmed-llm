"""
clients/fl_client.py
Flower Federated Learning Client for our FedMed-LLM
One instance of this runs per hospital node (A, B, C)

Usage (Day 2):
    python clients/fl_client.py --node hospital_a --server-address 127.0.0.1:8080
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import flwr as fl

# ── Config ────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/phi-2"
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LENGTH   = 256
BATCH_SIZE   = 1
GRAD_ACCUM   = 4
LEARNING_RATE = 2e-4
LOCAL_EPOCHS  = 1   # Each client trains 1 epoch per FL round

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def load_model_and_tokenizer():
    """Load Phi-2 with 4-bit quantisation and LoRA adapters."""
    print(f"  Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading Phi-2 in 4-bit (takes 2-3 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    return model, tokenizer


def load_node_data(node_name: str, tokenizer):
    """Load this node's local partition and tokenise."""
    data_path = os.path.join(DATA_DIR, f"{node_name}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Run 'python scripts/prepare_data.py' first."
        )

    with open(data_path) as f:
        records = json.load(f)

    print(f"  Loaded {len(records)} QA pairs for {node_name}")

    def format_and_tokenize(example):
        text = f"Question: {example['question']}\nAnswer: {example['answer']}"
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    dataset = Dataset.from_list(records)
    tokenized = dataset.map(
        format_and_tokenize,
        remove_columns=["question", "answer"],
        desc="Tokenising",
    )
    return tokenized


def get_lora_weights(model):
    """Extract only the LoRA adapter weights as numpy arrays."""
    weights = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            weights.append(param.detach().cpu().float().numpy())
    return weights


def set_lora_weights(model, weights):
    """Set LoRA adapter weights from a list of numpy arrays."""
    lora_params = [
        (name, param)
        for name, param in model.named_parameters()
        if "lora" in name and param.requires_grad
    ]
    for (name, param), weight in zip(lora_params, weights):
        param.data = torch.tensor(weight, dtype=param.dtype).to(param.device)


# ── Flower Client ─────────────────────────────────────────────
class FedMedClient(fl.client.NumPyClient):
    def __init__(self, node_name: str):
        self.node_name = node_name
        print(f"\n[{node_name}] Initialising client...")
        self.model, self.tokenizer = load_model_and_tokenizer()
        self.train_data = load_node_data(node_name, self.tokenizer)
        print(f"[{node_name}] Client ready. Training samples: {len(self.train_data)}")

    def get_parameters(self, config):
        """Return current LoRA adapter weights to the server."""
        return get_lora_weights(self.model)

    def fit(self, parameters, config):
        """
        Receive global weights from server,
        run local training for LOCAL_EPOCHS,
        return updated weights + metrics.
        """
        round_num = config.get("server_round", "?")
        print(f"\n[{self.node_name}] Round {round_num}: Starting local training...")

        # Set received global weights
        set_lora_weights(self.model, parameters)

        # Training arguments
        output_dir = f"/tmp/fedmed_{self.node_name}_round{round_num}"
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=LOCAL_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_data,
        )

        result = trainer.train()
        train_loss = result.training_loss
        num_samples = len(self.train_data)

        print(f"[{self.node_name}] Round {round_num} complete. Loss: {train_loss:.4f}")

        return get_lora_weights(self.model), num_samples, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        """Evaluate the global model on local data (optional but good practice)."""
        set_lora_weights(self.model, parameters)
        # Simple perplexity estimate on first 100 samples
        sample = self.train_data.select(range(min(100, len(self.train_data))))
        args = TrainingArguments(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=BATCH_SIZE,
            report_to="none",
        )
        trainer = Trainer(model=self.model, args=args)
        metrics = trainer.evaluate(eval_dataset=sample)
        loss = metrics.get("eval_loss", 0.0)
        print(f"[{self.node_name}] Eval loss: {loss:.4f}")
        return loss, len(sample), {"eval_loss": loss}


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True,
                        choices=["hospital_a", "hospital_b", "hospital_c"],
                        help="Which hospital node this client represents")
    parser.add_argument("--server-address", default="127.0.0.1:8080",
                        help="FL server address (host:port)")
    args = parser.parse_args()

    client = FedMedClient(node_name=args.node)

    print(f"\n[{args.node}] Connecting to FL server at {args.server_address}...")
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )
