from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import yaml


class GeminiCloudConfig(TypedDict):
    model: str
    dimensions: int | None  # Matryoshka output dim; None = model default (3072)
    base_url: str


class EncoderConfig(TypedDict, total=False):
    backend: str
    embedding_backend: str  # "local" | "gemini_cloud"
    dimension: int
    sigma_mode: str
    low_rank: int
    on_mismatch: str  # "retrain" | "heuristic" | "error"
    gemini_cloud: GeminiCloudConfig


class ChunkingConfig(TypedDict):
    max_tokens: int
    overlap_tokens: int


class RetrievalConfig(TypedDict):
    ann_candidates: int
    top_k: int
    diversity_threshold: float
    metric: str


class GenerationConfig(TypedDict):
    mode: str
    provider: str
    model: str
    timeout_seconds: float


class ProviderEnvConfig(TypedDict):
    api_key: str
    base_url: str
    model: str


class RuntimeEnvConfig(TypedDict):
    app_env: str
    store_path: str
    top_k: int


class AmbiguityBuckets(TypedDict):
    medium_entropy: float
    high_entropy: float


class EvaluationConfig(TypedDict):
    ambiguity_buckets: AmbiguityBuckets


class ContinualLearningConfig(TypedDict):
    enabled: bool
    train_every_n: int


class CloudflareClefConfig(TypedDict):
    model: str
    account_id_env: str
    api_token_env: str
    timeout_seconds: float
    max_retries: int


class DecisionEngineConfig(TypedDict, total=False):
    provider: str
    cloudflare: CloudflareClefConfig


class ProjectConfig(TypedDict, total=False):
    encoder: EncoderConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    continual_learning: ContinualLearningConfig
    decision_engine: DecisionEngineConfig


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config payload must be a mapping")
    return ProjectConfig(**payload)


def load_dotenv_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        values[key.strip()] = cleaned
    return values


def load_provider_env(path: str | Path = ".env") -> ProviderEnvConfig:
    file_values = load_dotenv_file(path)
    if str(path) != ".env" and file_values.get("MISTRAL_API_KEY"):
        api_key = file_values["MISTRAL_API_KEY"]
    else:
        api_key = os.environ.get("MISTRAL_API_KEY") or file_values.get(
            "MISTRAL_API_KEY", ""
        )
    base_url = (
        file_values.get("MISTRAL_BASE_URL")
        if str(path) != ".env" and "MISTRAL_BASE_URL" in file_values
        else (os.environ.get("MISTRAL_BASE_URL") or file_values.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions"))
    )
    model = (
        file_values.get("MISTRAL_MODEL")
        if str(path) != ".env" and "MISTRAL_MODEL" in file_values
        else (os.environ.get("MISTRAL_MODEL") or file_values.get("MISTRAL_MODEL", "mistral-small-latest"))
    )
    if not api_key:
        raise ValueError(
            "Missing Mistral API key. Set MISTRAL_API_KEY or add MISTRAL_API_KEY to .env."
        )
    return ProviderEnvConfig(api_key=api_key, base_url=base_url, model=model)


def load_runtime_env(path: str | Path = ".env") -> RuntimeEnvConfig:
    file_values = load_dotenv_file(path)
    app_env = os.environ.get("GAUSSIAN_RAG_ENV") or file_values.get(
        "GAUSSIAN_RAG_ENV", "development"
    )
    store_path = os.environ.get("GAUSSIAN_RAG_STORE_PATH") or file_values.get(
        "GAUSSIAN_RAG_STORE_PATH", ".gaussian_rag_store"
    )
    top_k_raw = os.environ.get("GAUSSIAN_RAG_TOP_K") or file_values.get(
        "GAUSSIAN_RAG_TOP_K", "5"
    )
    return RuntimeEnvConfig(
        app_env=app_env,
        store_path=store_path,
        top_k=max(int(top_k_raw), 1),
    )


def load_embedding_env(path: str | Path = ".env") -> dict[str, str]:
    """Return GEMINI_API_KEY from env or .env file.

    Returns an empty dict when the key is absent — callers should raise only
    when embedding_backend == "gemini_cloud" and the key is missing.
    """
    file_values = load_dotenv_file(path)
    api_key = os.environ.get("GEMINI_API_KEY") or file_values.get("GEMINI_API_KEY", "")
    return {"api_key": api_key} if api_key else {}


class CloudflareEnvConfig(TypedDict):
    account_id: str
    api_token: str


def load_cloudflare_env(path: str | Path = ".env") -> CloudflareEnvConfig:
    """Return Cloudflare account_id and api_token from environment or .env file.

    Raises ValueError if either credential is missing — ClefClient requires both.
    """
    file_values = load_dotenv_file(path)
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or file_values.get(
        "CLOUDFLARE_ACCOUNT_ID", ""
    )
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN") or file_values.get(
        "CLOUDFLARE_API_TOKEN", ""
    )
    if not account_id:
        raise ValueError(
            "Missing Cloudflare account ID. Set CLOUDFLARE_ACCOUNT_ID or add it to .env."
        )
    if not api_token:
        raise ValueError(
            "Missing Cloudflare API token. Set CLOUDFLARE_API_TOKEN or add it to .env."
        )
    return CloudflareEnvConfig(account_id=account_id, api_token=api_token)
