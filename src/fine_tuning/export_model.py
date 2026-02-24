"""
Export trained model for inference
Merges LoRA weights and optionally quantizes to GGUF
"""

from unsloth import FastLanguageModel
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_model(
    model_path: str,
    output_path: str,
    quantization: str = "q5_k_m"
):
    """
    Export fine-tuned model
    
    Args:
        model_path: Path to LoRA checkpoint
        output_path: Output directory
        quantization: GGUF quantization method (q5_k_m, q8_0, f16, etc.)
    """
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model with LoRA weights
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,  # Load in FP16 for merging
    )
    
    # Merge LoRA weights into base model
    logger.info("Merging LoRA weights...")
    model = FastLanguageModel.for_inference(model)  # Optimizes for inference
    
    # Save merged model
    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_16bit"  # or "merged_4bit" for smaller size
    )
    
    # Optionally export to GGUF for Ollama
    if quantization:
        logger.info(f"Exporting to GGUF ({quantization} quantization)...")
        model.save_pretrained_gguf(
            f"{output_path}/gguf",
            tokenizer,
            quantization_method=quantization
        )
        logger.info(f"GGUF saved to: {output_path}/gguf")
    
    logger.info("✅ Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--quantization", default="q5_k_m", help="GGUF quantization")
    
    args = parser.parse_args()
    export_model(args.model, args.output, args.quantization)