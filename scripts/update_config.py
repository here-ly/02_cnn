"""Update YAML config values via CLI — used by kaggle.ps1."""
import argparse
import sys
from pathlib import Path

import yaml


def update_yaml(filepath, updates):
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for key, value in updates.items():
        if value is None:
            continue
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Updated {filepath}")


def show_current(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print(f"Current values in {filepath}:")
    for section in ["data", "optim", "train", "wandb"]:
        if section in config:
            for k, v in config[section].items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        print(f"  {section}.{k}.{sk} = {sv}")
                else:
                    print(f"  {section}.{k} = {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update YAML config")
    parser.add_argument("--file", required=True, help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", help="key=value pairs, use dot notation (e.g. optim.lr=0.001 train.epochs=50)")
    parser.add_argument("--show", action="store_true", help="Show current values")
    args = parser.parse_args()

    if args.show:
        show_current(args.file)

    updates = {}
    if args.set:
        for s in args.set:
            if "=" not in s:
                print(f"Warning: invalid format '{s}', expected key=value", file=sys.stderr)
                continue
            k, v = s.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if v.lower() in ("true", "false"):
                        v = v.lower() == "true"
                    elif v.lower() in ("none", "null"):
                        v = None
            updates[k] = v

    if updates:
        update_yaml(args.file, updates)
