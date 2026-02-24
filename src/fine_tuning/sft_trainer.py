"""
SFT Trainer for Crypto Trading Oracle
Fine-tunes Qwen2.5-32B on trading instruction data using Unsloth
"""

import os
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict):
    """Load model and tokenizer with Unsloth optimizations"""
    
    logger.info(f"Loading model: {config['model']['name']}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        dtype=None,  # Auto-detect (will use bfloat16 on Blackwell)
        load_in_4bit=config['model']['load_in_4bit'],
    )
    
    logger.info("Applying LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
        random_state=42,
    )
    
    return model, tokenizer


def formatting_func(example):
    """Format examples for training (Alpaca format)"""
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input'] if example['input'] else ''}

### Response:
{example['output']}"""
    return text


def prepare_dataset(config: dict, tokenizer):
    """Load and prepare dataset"""
    
    logger.info("Loading datasets...")
    
    # Load train dataset
    train_dataset = load_dataset(
        'json',
        data_files=config['dataset']['train_file'],
        split='train'
    )
    
    # Load eval dataset
    eval_dataset = load_dataset(
        'json',
        data_files=config['dataset']['eval_file'],
        split='train'
    )
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Eval examples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def train(config_path: str = "config/training_config.yaml"):
    """Main training function"""
    
    # Load config
    config = load_config(config_path)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        report_to="none",  # Can add wandb/tensorboard later
    )
    
    # Initialize trainer
    logger.info("Initializing SFT Trainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        formatting_func=formatting_func,
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
    )
    
    # Print GPU stats before training
    logger.info("=" * 70)
    logger.info("GPU Memory Before Training:")
    logger.info(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    logger.info("=" * 70)
    
    # Train!
    logger.info("🚀 Starting training...")
    trainer.train()
    
    # Print GPU stats after training
    logger.info("=" * 70)
    logger.info("GPU Memory After Training:")
    logger.info(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    logger.info("=" * 70)
    
    # Save final model
    logger.info("💾 Saving final model...")
    trainer.save_model(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    logger.info("✅ Training complete!")
    logger.info(f"Model saved to: {config['training']['output_dir']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config"
    )
    
    args = parser.parse_args()
    train(args.config)