"""Configuration loader — reads config/config.yaml and returns a plain dict."""
from pathlib import Path
import yaml


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config.  Defaults to <project_root>/config/config.yaml."""
    if path is None:
        # Walk up from this file to find project root (contains config/)
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "config" / "config.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("config/config.yaml not found in any parent directory.")
    return yaml.safe_load(Path(path).read_text())
