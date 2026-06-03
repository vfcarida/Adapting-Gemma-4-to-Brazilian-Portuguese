from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger

__all__ = ["load_config", "get_logger"]


def set_seed(seed: int = 42):
    """Lazy import to avoid torch dependency at module level."""
    from src.utils.seed import set_seed as _set_seed

    _set_seed(seed)
