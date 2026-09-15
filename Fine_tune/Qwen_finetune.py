"""
QLoRA domain adaptation of Qwen on a directory of physics papers.

Pipeline
--------
1. Check the corresponding Ollama model.
2. Load the matching Hugging Face Qwen checkpoint.
3. Extract text from PDF physics papers.
4. Clean the extracted text.
5. Tokenize continuously.
6. Pack tokens into fixed-length blocks.
7. Load Qwen in 4-bit NF4.
8. Add LoRA adapters.
9. Train with causal language modeling.
10. Save LoRA adapter.
11. Optionally merge LoRA into full Hugging Face model.
12. Convert merged model to GGUF separately with llama.cpp.
"""

import os
import re
import json
import math
import random
import subprocess
from pathlib import Path
from typing import List, Dict

import torch

from pypdf import PdfReader

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)


# ==============================================================
# Configuration
# ==============================================================

# Name of your model in Ollama.
OLLAMA_MODEL = "qwen2.5:7b"

# IMPORTANT:
# This must correspond to your Ollama model.
#
# Examples:
# Qwen/Qwen2.5-1.5B-Instruct
# Qwen/Qwen2.5-3B-Instruct
# Qwen/Qwen2.5-7B-Instruct
# Qwen/Qwen3-4B
# Qwen/Qwen3-8B
#
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PAPERS_DIR = Path("./papers")

OUTPUT_DIR = Path("./output")

ADAPTER_DIR = OUTPUT_DIR / "qwen_physics_lora"

MERGED_DIR = OUTPUT_DIR / "qwen_physics_merged"

DATA_CACHE = OUTPUT_DIR / "physics_dataset"

# Context/chunk size.
#
# 2048 is a good starting point for modest GPUs.
# Increase to 4096 if your GPU can comfortably handle it.
MAX_SEQ_LENGTH = 2048

# QLoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 2

PER_DEVICE_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 16

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03

LOGGING_STEPS = 10

SAVE_STEPS = 200

SEED = 42

# Validation fraction
TEST_SIZE = 0.02


# ==============================================================
# Utilities
# ==============================================================

def print_gpu_info():

    print("=" * 60)
    print("GPU information")
    print("=" * 60)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("GPU:", torch.cuda.get_device_name(0))

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"VRAM: {total_memory:.2f} GB")

        print(
            "bfloat16 supported:",
            torch.cuda.is_bf16_supported()
        )

    print()


def inspect_ollama_model(model_name):

    print("=" * 60)
    print("Checking Ollama model")
    print("=" * 60)

    try:

        result = subprocess.run(
            [
                "ollama",
                "show",
                model_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout)

    except FileNotFoundError:

        print(
            "WARNING: Ollama executable was not found."
        )

    except subprocess.CalledProcessError as e:

        print(
            "WARNING: Could not inspect Ollama model."
        )

        print(e.stderr)


# ==============================================================
# PDF extraction
# ==============================================================

def clean_pdf_text(text: str) -> str:
    """
    Perform conservative PDF cleanup.

    Don't over-clean scientific text because equations,
    symbols and technical formatting may contain useful
    information.
    """

    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", " ")

    # Fix common hyphenated line breaks:
    #
    # "trans-\nformer" -> "transformer"
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )

    # Convert remaining newlines to spaces
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text,
    )

    # Collapse excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Collapse repeated whitespace
    text = re.sub(
        r"\s{2,}",
        " ",
        text,
    )

    return text.strip()


def extract_pdf(path: Path) -> str:

    print(f"Reading: {path.name}")

    try:

        reader = PdfReader(str(path))

    except Exception as e:

        print(
            f"Could not open {path}: {e}"
        )

        return ""

    pages = []

    for page_number, page in enumerate(reader.pages):

        try:

            text = page.extract_text()

        except Exception as e:

            print(
                f"  page {page_number}: extraction error: {e}"
            )

            continue

        if text:

            pages.append(text)

    full_text = "\n".join(pages)

    return clean_pdf_text(full_text)


def load_all_papers(directory: Path) -> List[Dict]:

    pdf_files = sorted(
        directory.rglob("*.pdf")
    )

    print(
        f"\nFound {len(pdf_files)} PDF papers."
    )

    documents = []

    for pdf in pdf_files:

        text = extract_pdf(pdf)

        # Ignore obviously broken extraction
        if len(text) < 1000:

            print(
                f"Skipping {pdf.name}: too little extracted text."
            )

            continue

        documents.append(
            {
                "source": pdf.name,
                "text": text,
            }
        )

    print(
        f"\nSuccessfully extracted {len(documents)} papers."
    )

    return documents


# ==============================================================
# Dataset preparation
# ==============================================================

def create_token_blocks(
    documents,
    tokenizer,
    block_size,
):
    """
    Tokenize complete papers and concatenate them.

    Then split the continuous token stream into fixed-size
    language-modeling blocks.

    This is preferable to naïvely splitting every N characters,
    because the model actually works with tokens.
    """

    all_input_ids = []

    eos_id = tokenizer.eos_token_id

    for i, doc in enumerate(documents):

        print(
            f"Tokenizing {i + 1}/{len(documents)}: "
            f"{doc['source']}"
        )

        tokenized = tokenizer(
            doc["text"],
            add_special_tokens=False,
            truncation=False,
        )

        ids = tokenized["input_ids"]

        all_input_ids.extend(ids)

        # Separate papers explicitly.
        if eos_id is not None:
            all_input_ids.append(eos_id)

    print()
    print(
        f"Total tokens: {len(all_input_ids):,}"
    )

    examples = []

    for start in range(
        0,
        len(all_input_ids) - block_size + 1,
        block_size,
    ):

        block = all_input_ids[
            start:start + block_size
        ]

        examples.append(
            {
                "input_ids": block,
                "attention_mask": [1] * len(block),
                "labels": block.copy(),
            }
        )

    print(
        f"Created {len(examples):,} training sequences "
        f"of {block_size} tokens."
    )

    return Dataset.from_list(examples)


# ==============================================================
# Model
# ==============================================================

def create_qlora_model(model_name):

    print()
    print("=" * 60)
    print("Loading Qwen in 4-bit")
    print("=" * 60)

    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):

        compute_dtype = torch.bfloat16

        print(
            "Using bfloat16 computation."
        )

    else:

        compute_dtype = torch.float16

        print(
            "Using float16 computation."
        )

    quantization_config = BitsAndBytesConfig(

        # Core QLoRA setting
        load_in_4bit=True,

        # NF4 is recommended for QLoRA training
        bnb_4bit_quant_type="nf4",

        # Quantize the quantization constants as well
        bnb_4bit_use_double_quant=True,

        # Matrix multiplication precision
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(

        model_name,

        quantization_config=quantization_config,

        device_map="auto",

        torch_dtype=compute_dtype,

        trust_remote_code=False,
    )

    # Needed before LoRA is attached to a quantized model.
    model = prepare_model_for_kbit_training(
        model
    )

    # Important for memory reduction during training.
    model.gradient_checkpointing_enable()

    model.config.use_cache = False

    # Qwen transformer layers normally contain these modules.
    #
    # Applying LoRA to all major attention + MLP projections
    # gives stronger domain adaptation than q_proj/v_proj alone.
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type="CAUSAL_LM",

        target_modules=target_modules,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print()
    model.print_trainable_parameters()

    return model


# ==============================================================
# Main training
# ==============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print_gpu_info()

    # ----------------------------------------------------------
    # 1. Inspect Ollama installation
    # ----------------------------------------------------------

    inspect_ollama_model(
        OLLAMA_MODEL
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        f"Ollama model = {OLLAMA_MODEL}"
    )

    print(
        f"Hugging Face training model = {HF_MODEL}"
    )

    print(
        "Verify these represent the SAME base model before training."
    )

    # ----------------------------------------------------------
    # 2. Tokenizer
    # ----------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL,
        use_fast=True,
        trust_remote_code=False,
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    # ----------------------------------------------------------
    # 3. Papers
    # ----------------------------------------------------------

    documents = load_all_papers(
        PAPERS_DIR
    )

    if not documents:

        raise RuntimeError(
            "No usable PDFs were found."
        )

    # ----------------------------------------------------------
    # 4. Tokenized LM dataset
    # ----------------------------------------------------------

    dataset = create_token_blocks(

        documents=documents,

        tokenizer=tokenizer,

        block_size=MAX_SEQ_LENGTH,
    )

    split_dataset = dataset.train_test_split(

        test_size=TEST_SIZE,

        seed=SEED,

        shuffle=True,
    )

    train_dataset = split_dataset["train"]

    eval_dataset = split_dataset["test"]

    print()
    print(
        f"Train sequences: {len(train_dataset):,}"
    )

    print(
        f"Evaluation sequences: {len(eval_dataset):,}"
    )

    # ----------------------------------------------------------
    # 5. QLoRA model
    # ----------------------------------------------------------

    model = create_qlora_model(
        HF_MODEL
    )

    # The inputs already contain labels.
    #
    # mlm=False means ordinary causal language modeling,
    # not BERT-style masked language modeling.
    data_collator = DataCollatorForLanguageModeling(

        tokenizer=tokenizer,

        mlm=False,
    )

    # ----------------------------------------------------------
    # 6. Training configuration
    # ----------------------------------------------------------

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    args = TrainingArguments(

        output_dir=str(
            OUTPUT_DIR / "checkpoints"
        ),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_BATCH_SIZE
        ),

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_ratio=WARMUP_RATIO,

        # Memory-efficient optimizer commonly used with QLoRA
        optim="paged_adamw_8bit",

        bf16=use_bf16,

        fp16=not use_bf16,

        gradient_checkpointing=True,

        logging_steps=LOGGING_STEPS,

        save_steps=SAVE_STEPS,

        eval_strategy="steps",

        eval_steps=SAVE_STEPS,

        save_total_limit=2,

        report_to="none",

        remove_unused_columns=False,

        dataloader_pin_memory=True,

        seed=SEED,
    )

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        data_collator=data_collator,
    )

    # ----------------------------------------------------------
    # 7. Train
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("Starting QLoRA training")
    print("=" * 60)

    trainer.train()

    # ----------------------------------------------------------
    # 8. Evaluation
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("Evaluating")
    print("=" * 60)

    metrics = trainer.evaluate()

    print(metrics)

    eval_loss = metrics.get(
        "eval_loss"
    )

    if eval_loss is not None:

        try:

            perplexity = math.exp(
                eval_loss
            )

            print(
                f"Perplexity: {perplexity:.3f}"
            )

        except OverflowError:

            pass

    # ----------------------------------------------------------
    # 9. Save QLoRA adapter
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("Saving LoRA adapter")
    print("=" * 60)

    ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        ADAPTER_DIR
    )

    metadata = {

        "ollama_model":
            OLLAMA_MODEL,

        "hf_base_model":
            HF_MODEL,

        "max_seq_length":
            MAX_SEQ_LENGTH,

        "lora_r":
            LORA_R,

        "lora_alpha":
            LORA_ALPHA,

        "epochs":
            NUM_EPOCHS,
    }

    with open(
        ADAPTER_DIR / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Adapter saved to {ADAPTER_DIR}"
    )

    print()
    print(
        "Training finished."
    )


if __name__ == "__main__":
    main()
