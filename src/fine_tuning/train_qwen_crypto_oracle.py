"""
Fine-tune Qwen2.5-32B-Instruct for Crypto Trading Oracle
Using Unsloth for 2-5x faster training!
Optimized for NVIDIA Grace-Blackwell GB10 (128GB)
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
if False:  # imported conditionally below when use_wandb=True
    import wandb
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model - Use pre-quantized Unsloth version!
    model_name = "unsloth/qwen2.5-32b-instruct-bnb-4bit"

    # Data
    train_data = "datasets/enhanced_sft_train.jsonl"
    eval_data = "datasets/enhanced_sft_eval.jsonl"

    # Output
    output_dir = "models/crypto-oracle-qwen-32b"

    # Staged training - splits dataset into N chunks trained sequentially.
    # Each chunk resumes from the previous stage's checkpoint.
    # Increase num_stages if VRAM is still too high.
    num_stages = 4

    # Training - reduced batch size to control VRAM; accumulation preserves
    # effective batch size of 16 (1 * 16 = 16, same as before 8 * 2 = 16)
    num_train_epochs = 3
    per_device_train_batch_size = 1   # was 8 — primary VRAM reduction
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 16  # was 2 — keeps effective batch = 16
    learning_rate = 2e-4
    max_seq_length = 2048

    # LoRA settings — reduced rank saves adapter memory
    lora_r = 16    # was 64
    lora_alpha = 16
    lora_dropout = 0.05

    # Optimization
    optim = "adamw_8bit"
    warmup_steps = 50
    logging_steps = 10
    save_steps = 200
    eval_steps = 200
    save_total_limit = 3

    # Mixed precision
    bf16 = True

    # Weights & Biases (optional)
    use_wandb = False
    wandb_project = "crypto-oracle"
    wandb_run_name = "qwen-32b-unsloth"


# ============================================================================
# CHAT TEMPLATE
# ============================================================================

chat_template = """<|im_start|>system
You are a professional cryptocurrency trading advisor with expertise in technical analysis, market sentiment, on-chain metrics, and risk management. Provide clear, actionable trading recommendations based on comprehensive market data.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""


def format_instruction(example):
    """Format for Qwen2.5 chat template"""
    return {
        "text": chat_template.format(
            example['instruction'],
            example['output']
        )
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    print("="*70)
    print("CRYPTO ORACLE - QWEN 2.5 32B FINE-TUNING (UNSLOTH)")
    print("="*70)
    print(f"\nDataset:    {Config.train_data}")
    print(f"Eval:       {Config.eval_data}")
    print(f"Model:      {Config.model_name}")
    print(f"Output:     {Config.output_dir}")
    print(f"Batch size: {Config.per_device_train_batch_size} x {Config.gradient_accumulation_steps} = {Config.per_device_train_batch_size * Config.gradient_accumulation_steps} effective")
    print(f"Stages:     {Config.num_stages}")
    print()

    if Config.use_wandb:
        import wandb
        wandb.init(
            project=Config.wandb_project,
            name=Config.wandb_run_name,
            config=vars(Config)
        )

    # ========================================================================
    # 1. LOAD & FORMAT DATASETS
    # ========================================================================

    print("Loading datasets...")

    train_dataset = load_dataset('json', data_files=Config.train_data, split='train')
    eval_dataset = load_dataset('json', data_files=Config.eval_data, split='train')

    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Eval:  {len(eval_dataset):,} examples")

    print("Formatting datasets...")
    train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_instruction, remove_columns=eval_dataset.column_names)

    # Split train dataset into stages
    total = len(train_dataset)
    stage_size = total // Config.num_stages
    stages = [
        train_dataset.select(range(i * stage_size, min((i + 1) * stage_size, total)))
        for i in range(Config.num_stages)
    ]
    print(f"  Split into {Config.num_stages} stages of ~{stage_size:,} examples each")

    # ========================================================================
    # 2. LOAD MODEL WITH RESUMPTION SUPPORT (UNSLOTH)
    # ========================================================================

    output_dir = Path(Config.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find last completed stage (adapter folder = completed)
    last_completed_stage = 0
    last_adapter_path = None
    for stage_num in range(Config.num_stages, 0, -1):
        stage_output_dir_check = output_dir / f"stage_{stage_num}"
        adapter_path = stage_output_dir_check / "adapter"
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            last_completed_stage = stage_num
            last_adapter_path = adapter_path
            break

    if last_completed_stage > 0:
        print(f"\n=== RESUMING TRAINING ===\nLast completed stage: {last_completed_stage}")
        print(f"Loading model from previous adapter: {last_adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(last_adapter_path),
            max_seq_length=Config.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        print(f"  Resumed model loaded — VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        start_stage_idx = last_completed_stage
    else:
        print(f"\nLoading model: {Config.model_name}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=Config.model_name,
            max_seq_length=Config.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        print(f"  Model loaded — VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        start_stage_idx = 0

    # ========================================================================
    # 3. ADD LORA ADAPTERS (only if starting fresh)
    # ========================================================================

    if last_completed_stage == 0:
        print("\nAdding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            lora_dropout=Config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    else:
        print("\nLoRA adapters already present from previous checkpoint.")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ========================================================================
    # 4. STAGED TRAINING LOOP (with resumption support)
    # ========================================================================

    print("\n" + "="*70)
    print("STARTING STAGED TRAINING")
    print("="*70)
    print(f"Starting from stage {start_stage_idx + 1}/{Config.num_stages}")

    for stage_idx in range(start_stage_idx, Config.num_stages):
        stage_data = stages[stage_idx]
        stage_num = stage_idx + 1
        stage_output_dir = output_dir / f"stage_{stage_num}"

        print(f"\n--- Stage {stage_num}/{Config.num_stages} | {len(stage_data):,} examples ---")

        # Intra-stage resume: use latest checkpoint- folder if it exists
        resume_from_checkpoint = None
        if stage_output_dir.exists():
            checkpoints = [
                d for d in stage_output_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
                resume_from_checkpoint = str(checkpoints[-1])
                print(f"  Resuming from previous checkpoint: {resume_from_checkpoint}")

        training_args = TrainingArguments(
            output_dir=str(stage_output_dir),
            num_train_epochs=Config.num_train_epochs,
            per_device_train_batch_size=Config.per_device_train_batch_size,
            per_device_eval_batch_size=Config.per_device_eval_batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            learning_rate=Config.learning_rate,
            warmup_steps=Config.warmup_steps,
            logging_steps=Config.logging_steps,
            save_steps=Config.save_steps,
            eval_steps=Config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=Config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=Config.bf16,
            optim=Config.optim,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="wandb" if Config.use_wandb else "none",
            run_name=f"{Config.wandb_run_name}-stage{stage_num}" if Config.use_wandb else None,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=stage_data,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=Config.max_seq_length,
            args=training_args,
            packing=False,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save adapter after each stage so we can resume if interrupted
        stage_adapter_path = stage_output_dir / "adapter"
        model.save_pretrained(stage_adapter_path)
        tokenizer.save_pretrained(stage_adapter_path)
        print(f"  Stage {stage_num} adapter saved to: {stage_adapter_path}")

        # Explicitly free trainer/optimizer memory between stages
        del trainer
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("ALL STAGES COMPLETE!")
    print("="*70)

    # ========================================================================
    # 5. SAVE FINAL MODEL
    # ========================================================================

    print("\nSaving final model...")

    final_model_path = output_dir / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"  Saved to: {final_model_path}")

    # GGUF conversion requires llama.cpp which needs Linux (apt-get).
    # Skip automatically on Windows; convert manually on a Linux machine if needed.
    import platform
    if platform.system() != "Windows":
        print("\nSaving GGUF...")
        gguf_path = output_dir / "gguf"
        gguf_path.mkdir(exist_ok=True)
        model.save_pretrained_gguf(
            str(gguf_path / "model"),
            tokenizer,
            quantization_method="q4_k_m"
        )
        print(f"  GGUF saved to: {gguf_path}")
    else:
        print("\nSkipping GGUF (Windows detected — llama.cpp requires Linux).")
        print("  To convert: copy final_model/ to a Linux box and run save_pretrained_gguf().")
    print("\nDone! Your Crypto Oracle is ready.")

    if Config.use_wandb:
        import wandb
        wandb.finish()


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def test_model(model_path: str, prompt: str):
    """Test the fine-tuned model"""
    
    print(f"Loading model from {model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)  # Enable inference mode
    
    # Format prompt
    formatted_prompt = chat_template.format(prompt, "")
    
    # Generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("\n" + "="*70)
    print("MODEL RESPONSE:")
    print("="*70)
    
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True
    )
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--test", type=str, help="Test model with prompt")
    parser.add_argument("--model", type=str, default="models/crypto-oracle-qwen-32b/final_model", help="Model path for testing")
    
    args = parser.parse_args()
    
    if args.train:
        train()
    elif args.test:
        test_model(args.model, args.test)
    else:
        print("Usage:")
        print("  Train: python train_qwen_crypto_oracle.py --train")
        print("  Test:  python train_qwen_crypto_oracle.py --test 'Analyze BTC/USD...'")