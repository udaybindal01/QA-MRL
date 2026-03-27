"""Logging utilities with Weights & Biases integration."""

import os
import logging
from typing import Any, Dict, Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def setup_logger(
    name: str = "qa-mrl",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with optional rich formatting and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    if HAS_RICH:
        console = Console()
        handler = RichHandler(console=console, show_path=False)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    return logger


class WandbLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                self.run = wandb.init(
                    project=config.get("logging", {}).get("wandb_project", "qa-mrl"),
                    entity=config.get("logging", {}).get("wandb_entity"),
                    config=config,
                )
            except Exception:
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self.wandb.finish()
