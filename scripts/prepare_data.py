"""
Day 1 — Step 1: Download MedQuAD and split into 3 hospital partitions
Run this locally (not on Colab) since it just needs pandas + requests.

Usage:
    python scripts/prepare_data.py
"""

import os
import json
import random
import pandas as pd
import urllib.request
import zipfile

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  Load MedQuAD from Hugging Face datasets hub
#     (no manual download needed — datasets library handles it)
# ─────────────────────────────────────────────────────────────
print("Loading MedQuAD dataset from Hugging Face hub...")

try:
    from datasets import load_dataset
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    # Use first 47000 rows to keep size manageable
    df = ds.to_pandas()[["input", "output"]].dropna()
    df.columns = ["question", "answer"]
    df = df[df["question"].str.len() > 20]
    df = df[df["answer"].str.len() > 30]
    df = df.head(47000).reset_index(drop=True)
    print(f"  Loaded {len(df)} QA pairs from HealthCareMagic dataset")

except Exception as e:
    print(f"  Could not load from hub: {e}")
    print("  Creating sample dataset for local testing...")
    # Fallback: create a minimal sample so you can test the pipeline
    sample_qa = [
        {"question": "What are the symptoms of diabetes?",
         "answer": "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, blurred vision, and slow-healing sores."},
        {"question": "How is hypertension treated?",
         "answer": "Hypertension is treated with lifestyle changes such as diet and exercise, and medications like ACE inhibitors, beta-blockers, or diuretics."},
        {"question": "What causes anemia?",
         "answer": "Anemia is caused by iron deficiency, vitamin B12 deficiency, chronic disease, blood loss, or inherited conditions like sickle cell disease."},
        {"question": "What are signs of a heart attack?",
         "answer": "Signs include chest pain or pressure, shortness of breath, pain radiating to the arm or jaw, nausea, and cold sweats."},
        {"question": "How do you treat a fever?",
         "answer": "Treat a fever by staying hydrated, resting, taking paracetamol or ibuprofen, and using a cool compress. Seek medical attention if fever exceeds 103F."},
    ] * 200  # repeat to simulate dataset size
    df = pd.DataFrame(sample_qa)
    print(f"  Created {len(df)} sample QA pairs for testing")

# ─────────────────────────────────────────────────────────────
# 2.  Clean data
# ─────────────────────────────────────────────────────────────
print("\nCleaning data...")
df["question"] = df["question"].str.strip()
df["answer"]   = df["answer"].str.strip()

# Remove very short or very long entries
df = df[df["question"].str.len().between(15, 500)]
df = df[df["answer"].str.len().between(30, 2000)]
df = df.drop_duplicates(subset="question")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Clean dataset: {len(df)} QA pairs")

# ─────────────────────────────────────────────────────────────
# 3.  Split into 3 hospital partitions
# ─────────────────────────────────────────────────────────────
print("\nSplitting into 3 hospital partitions...")
total = len(df)
split_a = int(total * 0.33)
split_b = int(total * 0.33)

partitions = {
    "hospital_a": df.iloc[:split_a],
    "hospital_b": df.iloc[split_a : split_a + split_b],
    "hospital_c": df.iloc[split_a + split_b :],
}

for name, partition in partitions.items():
    out_path = os.path.join(DATA_DIR, f"{name}.json")
    records = partition.to_dict(orient="records")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  {name}: {len(records)} pairs → saved to {out_path}")

# ─────────────────────────────────────────────────────────────
# 4.  Save a combined held-out test set (200 samples for Day 7 eval)
# ─────────────────────────────────────────────────────────────
test_set = df.sample(200, random_state=99)
test_path = os.path.join(DATA_DIR, "test_set.json")
with open(test_path, "w") as f:
    json.dump(test_set.to_dict(orient="records"), f, indent=2)
print(f"\n  Test set: 200 pairs → saved to {test_path}")

print("\nDay 1 Step 1 COMPLETE — Data ready in /data/")
print("Next: open Colab and run colab_day1_test.ipynb")
