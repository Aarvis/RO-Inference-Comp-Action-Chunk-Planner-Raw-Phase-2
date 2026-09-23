from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .config import JaxRuntimeConfig


logger = logging.getLogger(__name__)

PALIGEMMA_TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"


def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return module_root().parent


def vendor_root() -> Path:
    return project_root() / "vendor" / "openpi" / "src"


def default_tokenizer_path() -> Path:
    local = module_root() / "assets" / "paligemma_tokenizer.model"
    if local.is_file():
        return local
    legacy = module_root().parents[1] / "RO-Inference" / "OpenPI_Module" / "assets" / "paligemma_tokenizer.model"
    return legacy if legacy.is_file() else local


def configure_jax_environment(config: JaxRuntimeConfig) -> None:
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        logger.warning(
            "JAX appears to be imported already. XLA memory environment variables may no longer take effect."
        )
    if config.preallocate is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true" if config.preallocate else "false")
    if config.mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.mem_fraction))
    if config.allocator is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", str(config.allocator))
    if config.platforms is not None:
        os.environ.setdefault("JAX_PLATFORMS", str(config.platforms))


def ensure_openpi_on_path(openpi_source_root: str | Path | None = None) -> None:
    # A Phase-2 bundle may deliberately select a frozen H25/S1 source profile
    # instead of this project's default H15/S2 source. Honor that explicit
    # selection before falling back to the default vendor location.
    if openpi_source_root not in (None, ""):
        source = Path(openpi_source_root).resolve()
        if not (source / "openpi").is_dir():
            raise FileNotFoundError(f"Configured OpenPI source root is invalid: {source}")
        root = str(source)
        if root not in sys.path:
            sys.path.insert(0, root)
        return

    vendored = vendor_root()
    if (vendored / "openpi").is_dir():
        root = str(vendored)
        if root not in sys.path:
            sys.path.insert(0, root)
        return

    legacy_vendored = module_root() / "vendor"
    if (legacy_vendored / "openpi").is_dir():
        root = str(legacy_vendored)
        if root not in sys.path:
            sys.path.insert(0, root)
        return



def patch_paligemma_tokenizer_download(
    download_module: Any,
    tokenizer_model_path: str | Path | None,
) -> Path:
    tokenizer_path = Path(tokenizer_model_path).resolve() if tokenizer_model_path else default_tokenizer_path().resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            "PaliGemma tokenizer model not found. Set paths.tokenizer_model_path or package "
            f"the tokenizer at {module_root() / 'assets' / 'paligemma_tokenizer.model'}."
        )

    original: Callable[..., Path] = getattr(
        download_module,
        "_origami_comp_action_chunk_original_maybe_download",
        download_module.maybe_download,
    )
    setattr(download_module, "_origami_comp_action_chunk_original_maybe_download", original)

    def maybe_download(url: str, *args: Any, **kwargs: Any) -> Path:
        if str(url) == PALIGEMMA_TOKENIZER_URI:
            return tokenizer_path
        return original(url, *args, **kwargs)

    download_module.maybe_download = maybe_download
    os.environ.setdefault("RO_OPENPI_TOKENIZER_PATH", str(tokenizer_path))
    return tokenizer_path
