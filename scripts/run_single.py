"""
run_single.py — Run one model × one dataset experiment.

Usage
-----
# Auto-discover model and dataset configs (no --config_files needed)
python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo

# Explicitly specify config files (overrides auto-discovery)
python scripts/run_single.py \
    --model SASRec \
    --dataset amazon-videogames-2023-5c-llo \
    --config_files configs/models/SASRec.yaml configs/datasets/amazon-videogames-2023-5c-llo.yaml

# Local smoke-test (CPU, 2 epochs, uni100 eval, 200-user subset)
python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo-tiny

# Override any param inline (highest priority)
python scripts/run_single.py \
    --model SASRec \
    --dataset amazon-videogames-2023-5c-llo \
    --params learning_rate=0.005 n_layers=3

Config merge order (each layer overrides the previous):
  1. configs/base.yaml                      (always loaded)
  2. configs/models/{MODEL}.yaml            (auto-discovered, if exists)
  3. configs/datasets/{DATASET}.yaml        (auto-discovered, if exists)
     — OR —
     --config_files ...                     (explicit files, replaces steps 2-3)
  4. --params key=value ...                 (highest priority, inline overrides)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `src/` is importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed, set_color
from recbole.utils.enum_type import ModelType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_inline_params(param_list: list[str]) -> dict:
    """Convert ['key=value', ...] into a dict, casting types automatically."""
    result = {}
    for item in param_list:
        if "=" not in item:
            raise ValueError(
                f"--params entry must be in key=value format, got: {item!r}"
            )
        key, _, raw_value = item.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        result[key] = _cast(raw_value)
    return result


def _cast(value: str):
    """Try to cast a string to int, float, bool, or leave as str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def resolve_config_files(args) -> list[str]:
    """
    Build the final ordered list of YAML config file paths.

    Order:
      1. configs/base.yaml                    (always first)
      2a. configs/models/{MODEL}.yaml         (auto-discovered, if --config_files not given)
      2b. configs/datasets/{DATASET}.yaml     (auto-discovered, if --config_files not given)
         — OR —
         --config_files ...                   (explicit list, replaces 2a + 2b)
    """
    base = PROJECT_ROOT / "configs" / "base.yaml"
    if not base.exists():
        raise FileNotFoundError(f"Base config not found: {base}")

    paths = [str(base)]

    if args.config_files:
        # Explicit mode: user supplied their own list — use it as-is.
        for cf in args.config_files:
            p = Path(cf)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {p}")
            paths.append(str(p))
    else:
        # Auto-discovery mode: look for model and dataset configs by name.
        model_cfg = PROJECT_ROOT / "configs" / "models" / f"{args.model}.yaml"
        if model_cfg.exists():
            paths.append(str(model_cfg))
        else:
            print(f"  [info] No model config found at {model_cfg.relative_to(PROJECT_ROOT)}, skipping.")

        dataset_cfg = PROJECT_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml"
        if dataset_cfg.exists():
            paths.append(str(dataset_cfg))
        else:
            print(f"  [info] No dataset config found at {dataset_cfg.relative_to(PROJECT_ROOT)}, skipping.")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a single RecBole experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", required=True,
        help="RecBole model name, e.g. SASRec, BPR, LightGCN",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Dataset name matching a folder under dataset/, e.g. amazon-videogames-2023-5c-llo",
    )
    p.add_argument(
        "--config_files", nargs="*", default=[],
        metavar="FILE",
        help=(
            "Extra YAML config files to merge (space-separated). "
            "Paths relative to the project root are accepted. "
            "Later files override earlier ones. base.yaml is always prepended."
        ),
    )
    p.add_argument(
        "--params", nargs="*", default=[],
        metavar="KEY=VALUE",
        help=(
            "Inline config overrides with highest priority, e.g. "
            "--params learning_rate=0.005 n_layers=3"
        ),
    )
    p.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging for this run (e.g. for quick smoke tests).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Resolve config file list -------------------------------------------
    config_file_list = resolve_config_files(args)

    # --- Parse inline overrides ---------------------------------------------
    config_dict = parse_inline_params(args.params)

    # Inject dataset into config_dict so it always wins over any stale value
    # in a config file.
    config_dict["dataset"] = args.dataset

    # --- Print summary -------------------------------------------------------
    print("=" * 60)
    print(f"  model   : {args.model}")
    print(f"  dataset : {args.dataset}")
    print(f"  configs : ")
    for cf in config_file_list:
        try:
            rel = Path(cf).relative_to(PROJECT_ROOT)
        except ValueError:
            rel = cf
        print(f"            {rel}")
    if config_dict:
        print(f"  overrides: {config_dict}")
    print("=" * 60)

    # --- Run (mirrors run_recbole but skips get_flops, which crashes when
    #         pre-built item_id_list sequences exceed MAX_ITEM_LIST_LENGTH) ---
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        import json
        safe_cfg = {}
        for k in config:
            v = config[k]
            try:
                json.dumps(v)
                safe_cfg[k] = v
            except (TypeError, ValueError):
                safe_cfg[k] = str(v)
        wandb.init(
            project="rec-framework",
            name=f"{args.model}-{args.dataset}",
            config=safe_cfg,
        )

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    trainer = get_trainer(config["model_type"], config["model"])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    if use_wandb:
        wandb.log({"valid/" + k: v for k, v in best_valid_result.items()})
        wandb.log({"test/" + k: v for k, v in test_result.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
