# Fine-Tuning Guide for Movie Trailer Generation Model

This guide provides step-by-step instructions for fine-tuning a generative model to create movie trailers from scripts. Two strategies are provided: one for **RTX 2080 Laptop (8GB VRAM)** and another for **AWS Instance (64GB VRAM)**.

---

## Table of Contents
1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Fine-Tuning Strategies](#fine-tuning-strategies)
   - [RTX 2080 Laptop (8GB VRAM)](#rtx-2080-laptop-8gb-vram)
   - [AWS Instance (64GB VRAM)](#aws-instance-64gb-vram)
4. [Running the Fine-Tuning Scripts](#running-the-fine-tuning-scripts)
5. [Saving and Using the Fine-Tuned Model](#saving-and-using-the-fine-tuned-model)
6. [Troubleshooting](#troubleshooting)

---

## Overview

Fine-tuning involves adapting a pre-trained generative model (e.g., GPT-2, T5) to generate movie trailers from scripts. The process requires:
- A dataset of script-trailer pairs.
- A pre-trained model.
- Hardware with sufficient computational resources (GPU).

This guide provides two fine-tuning strategies tailored for different hardware setups.

---

## Hardware Requirements

### RTX 2080 Laptop (8GB VRAM)
- GPU: NVIDIA RTX 2080 (8GB VRAM)
- RAM: 16GB or higher
- Storage: SSD with at least 20GB free space
- Python 3.8+
- CUDA and cuDNN installed

### AWS Instance (64GB VRAM)
- GPU: NVIDIA A100 or similar (64GB VRAM)
- RAM: 128GB or higher
- Storage: SSD with at least 50GB free space
- Python 3.8+
- CUDA and cuDNN installed

---

## Fine-Tuning Strategies

### RTX 2080 Laptop (8GB VRAM)

#### Approach:
- Use a smaller model (e.g., **DistilGPT-2** or **T5-small**).
- Employ gradient accumulation and mixed precision training to optimize memory usage.
- Use a smaller batch size and truncate input sequences.

#### Script: `fine_tune_rtx2080.py`

```python
import torch
from transformers import DistilGPT2Tokenizer, DistilGPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset (script-trailer pairs)
dataset = load_dataset("your_dataset_name")  # Replace with your dataset

# Load tokenizer and model
tokenizer = DistilGPT2Tokenizer.from_pretrained("distilgpt2")
model = DistilGPT2LMHeadModel.from_pretrained("distilgpt2")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["script"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Small batch size for 8GB VRAM
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,  # Mixed precision training to save memory
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_model_rtx2080")
```

---

### AWS Instance (64GB VRAM)

#### Approach:
- Use a larger model (e.g., **GPT-3.5** or **T5-large**).
- Use a larger batch size and full sequence length.
- Employ **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.

#### Script: `fine_tune_aws.py`

```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load dataset (script-trailer pairs)
dataset = load_dataset("your_dataset_name")  # Replace with your dataset

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

# Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank adaptation
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"],  # Target specific layers
)

model = get_peft_model(model, lora_config)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["script"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,  # Larger batch size for 64GB VRAM
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    fp16=True,  # Mixed precision training
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_model_aws")
```

---

## Running the Fine-Tuning Scripts

1. **RTX 2080 Laptop**:
   - Save the script as `fine_tune_rtx2080.py`.
   - Run the script using:
     ```bash
     python fine_tune_rtx2080.py
     ```

2. **AWS Instance**:
   - Save the script as `fine_tune_aws.py`.
   - Run the script using:
     ```bash
     python fine_tune_aws.py
     ```

---

## Saving and Using the Fine-Tuned Model

- The fine-tuned model is saved in the `./fine_tuned_model_rtx2080` or `./fine_tuned_model_aws` directory.
- To use the model for inference:
  ```python
  from transformers import pipeline

  # Load the fine-tuned model
  generator = pipeline("text-generation", model="./fine_tuned_model_rtx2080")

  # Generate a trailer from a script
  script = "Your movie script here..."
  trailer = generator(script, max_length=512)
  print(trailer)
  ```

---

## Troubleshooting

1. **Out of Memory (OOM) Errors**:
   - Reduce the batch size or sequence length.
   - Enable gradient accumulation or mixed precision training.

2. **Slow Training**:
   - Use a more powerful GPU or distributed training.
   - Optimize data loading with `datasets` library.

3. **Model Not Converging**:
   - Increase the number of training epochs.
   - Adjust the learning rate or use a learning rate scheduler.

---

For further assistance, refer to the [Hugging Face documentation](https://huggingface.co/docs) or open an issue in the project repository.
