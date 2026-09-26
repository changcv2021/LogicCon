"""TOML settings; environment variables are expanded without reading credentials."""
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from .engine import MethodConfig
from .schema import require


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, str):
        value = os.path.expandvars(value)
        require(not re.search(r"\$\{[^}]+\}", value), f"Set required environment variable in: {value}")
    return value


@dataclass(frozen=True)
class BackendConfig:
    kind: str = "transformers"
    model: str = ""
    revision: str = "main"
    local_files_only: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 1536
    min_pixels: int = 200704
    max_pixels: int = 802816
    seed: int = 20260926
    base_url: str = ""
    api_key_env: str = "LOGICCON_API_KEY"
    timeout_seconds: int = 180
    transport_retries: int = 2

    def __post_init__(self):
        require(self.kind in {"transformers", "http"}, "backend.kind must be transformers or http")
        require(bool(self.model), "backend.model required")
        require(self.dtype in {"bfloat16", "float16", "float32"}, "Invalid dtype")
        require(self.device_map in {"auto", "cuda"}, "GPU inference requires device_map=auto or cuda")
        require(type(self.local_files_only) is bool, "local_files_only must be boolean")
        for key in ("max_new_tokens", "min_pixels", "max_pixels", "timeout_seconds"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"Invalid {key}")
        require(self.min_pixels <= self.max_pixels, "min_pixels > max_pixels")
        require(type(self.seed) is int and self.seed >= 0, "seed must be a nonnegative integer")
        require(type(self.transport_retries) is int and 0 <= self.transport_retries <= 5, "Invalid transport_retries")
        if self.kind == "http":
            from urllib.parse import urlparse
            url = urlparse(self.base_url)
            require(url.scheme in {"http", "https"} and url.netloc and not url.username
                    and not url.password and not url.query and not url.fragment, "Invalid base_url")
            require(bool(self.api_key_env), "api_key_env must name an environment variable")


@dataclass(frozen=True)
class RunConfig:
    backend: BackendConfig
    method: MethodConfig = field(default_factory=MethodConfig)

    def to_dict(self):
        return asdict(self)


def load_config(path, mode=None):
    raw = expand(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    require(set(raw) <= {"backend", "method"} and "backend" in raw, "Only [backend] and [method] are accepted")
    if mode:
        raw.setdefault("method", {})["mode"] = mode
    try:
        return RunConfig(BackendConfig(**raw["backend"]), MethodConfig(**raw.get("method", {})))
    except TypeError as exc:
        raise ValueError(f"Unknown or invalid config field: {exc}") from exc
