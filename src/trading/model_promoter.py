"""
Model Promoter — Priority 2 + 3

Manages versioned LoRA adapters and gates promotion behind an evaluation run.
A new adapter is only swapped into production if it matches or beats the current
adapter's pass rate on the standard benchmark scenarios.

Versioning layout:
  models/
    crypto-oracle-qwen-32b-v2/final_model/   ← original trained model
    adapters/
      20260512_081200/                        ← incremental retrains (timestamped)
      20260601_000000/
    crypto-oracle-lora-latest                ← symlink → current production adapter

On first run, the symlink is created pointing to the v2 final_model.
The paper trader and inference engine should use the symlink path so they
automatically pick up promotions without a service restart.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "models" / "adapters"
SYMLINK_PATH = PROJECT_ROOT / "models" / "crypto-oracle-lora-latest"
DEFAULT_PROD = PROJECT_ROOT / "models" / "crypto-oracle-qwen-32b-v2" / "final_model"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "fine_tuning" / "train_qwen_crypto_oracle.py"
EVAL_LOG_DIR = PROJECT_ROOT / "data" / "eval_logs"

_PROMOTION_TOLERANCE = 0.0   # candidate must match or beat current pass rate


# ============================================================================
# SYMLINK / VERSION MANAGEMENT
# ============================================================================

def get_production_adapter() -> Path:
    """
    Return the resolved path of the current production adapter.
    Creates the symlink pointing to v2 on first call if it doesn't exist.
    """
    if SYMLINK_PATH.exists() or SYMLINK_PATH.is_symlink():
        return SYMLINK_PATH.resolve()

    if DEFAULT_PROD.exists():
        _update_symlink(DEFAULT_PROD)
        logger.info(f"Initialised production symlink → {DEFAULT_PROD}")
        return DEFAULT_PROD

    raise FileNotFoundError(
        f"No production adapter found. Expected symlink at {SYMLINK_PATH} "
        f"or default model at {DEFAULT_PROD}."
    )


def _update_symlink(target: Path) -> None:
    """Atomically replace the production symlink (POSIX rename is atomic)."""
    tmp = SYMLINK_PATH.with_suffix(".tmp_link")
    if tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target.resolve())
    tmp.rename(SYMLINK_PATH)
    logger.info(f"Production symlink updated → {target}")


def _new_adapter_path() -> Path:
    """Return a new timestamped directory path inside ADAPTERS_DIR."""
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ADAPTERS_DIR / ts


# ============================================================================
# GPU THERMAL SAFETY
# ============================================================================

def _gpu_temp() -> int:
    """Return current GPU temperature in °C, or 0 if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


# ============================================================================
# EVALUATION
# ============================================================================

def _run_eval(model_path: Path) -> dict:
    """
    Run model_evaluator.py on the given adapter path.
    Returns the results dict (parsed from the saved JSON output file).
    """
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_LOG_DIR / f"eval_{model_path.name}_{ts}.json"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "testing" / "model_evaluator.py"),
        "--model",       str(model_path),
        "--temperature", "0.3",
        "--out",         str(out_path),
    ]
    logger.info(f"Evaluating {model_path.name} …")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"Eval process failed:\n{result.stderr[-2000:]}")
        return {"production_ready": False, "parsed": 0, "total_scenarios": 5}

    if out_path.exists():
        return json.loads(out_path.read_text())

    logger.error("Eval output file not written")
    return {"production_ready": False, "parsed": 0, "total_scenarios": 5}


def _pass_rate(eval_result: dict) -> float:
    n = eval_result.get("total_scenarios", 5)
    return eval_result.get("parsed", 0) / n if n > 0 else 0.0


# ============================================================================
# TRAINING + PROMOTION
# ============================================================================

def run_training_and_promote(feedback_path: Path = None, dry_run: bool = False) -> bool:
    """
    Full pipeline:
      1. GPU thermal check — skip if ≥75°C
      2. Run incremental fine-tuning → timestamped adapter directory
      3. Evaluate candidate vs current production
      4. Promote (update symlink) only if candidate ≥ current pass rate
      5. Write a promotion log to data/eval_logs/

    Returns True if a new adapter was promoted to production.
    """
    temp = _gpu_temp()
    if temp >= 75:
        logger.warning(f"GPU at {temp}°C — skipping training to avoid thermal throttle")
        return False

    current_path   = get_production_adapter()
    candidate_dir  = _new_adapter_path()
    # Training script writes to output_dir/final_model
    output_dir     = PROJECT_ROOT / "models" / f"crypto-oracle-incr-{candidate_dir.name}"
    candidate_path = output_dir / "final_model"

    logger.info(f"Starting incremental fine-tuning → {output_dir}")

    env = os.environ.copy()
    env["CRYPTO_ORACLE_OUTPUT_DIR"] = str(output_dir)

    if dry_run:
        logger.info(f"[DRY RUN] Would train → {output_dir} then eval vs {current_path.name}")
        return False

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--train"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if result.returncode != 0:
        logger.error("Training failed — production adapter unchanged")
        return False

    if not candidate_path.exists():
        logger.error(f"Expected adapter not found at {candidate_path}")
        return False

    # Evaluate both adapters on the standard benchmark
    logger.info("Running eval on candidate …")
    candidate_eval = _run_eval(candidate_path)
    logger.info("Running eval on current production …")
    current_eval   = _run_eval(current_path)

    c_rate = _pass_rate(candidate_eval)
    p_rate = _pass_rate(current_eval)
    promoted = c_rate >= p_rate - _PROMOTION_TOLERANCE

    logger.info(
        f"Candidate: {c_rate:.1%}  |  Current: {p_rate:.1%}  |  "
        f"{'PROMOTED' if promoted else 'REJECTED'}"
    )

    if promoted:
        _update_symlink(candidate_path)

    # Write promotion record
    promo_log = {
        "timestamp":            datetime.now().isoformat(),
        "candidate":            str(candidate_path),
        "current":              str(current_path),
        "candidate_pass_rate":  c_rate,
        "current_pass_rate":    p_rate,
        "promoted":             promoted,
    }
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (EVAL_LOG_DIR / f"promotion_{ts}.json").write_text(json.dumps(promo_log, indent=2))

    return promoted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status",  action="store_true", help="Show current production adapter")
    args = parser.parse_args()

    if args.status:
        try:
            prod = get_production_adapter()
            print(f"Production adapter: {prod}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
    else:
        promoted = run_training_and_promote(dry_run=args.dry_run)
        print(f"Promoted: {promoted}")
