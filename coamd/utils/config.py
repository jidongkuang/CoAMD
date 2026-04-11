from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--config', type=str, default=None, help='Optional YAML config file.')
    return parser


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()
    if config_args.config:
        config_path = Path(config_args.config)
        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**config)
    return parser.parse_args()
