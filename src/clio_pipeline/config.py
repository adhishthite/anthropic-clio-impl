"""Configuration management for the CLIO pipeline."""

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline settings, loaded from env vars and optionally overridden by a YAML config file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = ""
    azure_openai_api_key: str = ""
    jina_api_key: str = ""

    # Model config
    openai_model: str = "gpt-4.1-mini"
    azure_openai_deployment: str = ""
    openai_base_url: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_base_url: str = ""
    openai_temperature: float = 0.0
    client_max_retries: int = 4
    client_backoff_seconds: float = 1.0
    embedding_provider: str = "jina"
    embedding_model: str = "jina-embeddings-v3"

    # Clustering
    k_base_clusters: int = 20
    embedding_batch_size: int = 32
    cluster_label_sample_size: int = 12
    hierarchy_top_k: int = 10
    hierarchy_levels: int = 3
    hierarchy_target_group_size: int = 8
    viz_projection_method: str = "umap"
    random_seed: int = 42

    # Privacy thresholds
    min_unique_users: int = 5
    min_conversations_per_cluster: int = 10
    privacy_threshold_min_rating: int = 3
    privacy_audit_raw_sample_size: int = 40
    privacy_validation_enabled: bool = True

    # Evaluation
    eval_synthetic_count: int = 120
    eval_topic_count: int = 8
    eval_language_count: int = 5
    eval_seed: int = 19

    # Paths
    input_conversations_path: Path = Field(default=Path("data/mock/conversations_llm_200.jsonl"))
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("runs"))

    @classmethod
    def from_yaml(cls, config_path: str | Path, **overrides) -> "Settings":
        """Load settings from a YAML config file, with env vars and overrides applied on top."""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            yaml_config = {}
        merged = {**yaml_config, **overrides}
        return cls(**merged)

    def resolved_openai_base_url(self) -> str:
        """Resolve effective base URL, preferring explicit base URL then Azure endpoint."""

        candidate = (
            self.openai_base_url.strip()
            or self.azure_openai_base_url.strip()
            or self.azure_openai_endpoint.strip()
        )
        if not candidate:
            return ""

        normalized = candidate.rstrip("/")
        if "azure.com" in normalized.lower() and "openai/v1" not in normalized:
            normalized = f"{normalized}/openai/v1"
        return f"{normalized}/"

    def uses_azure_openai(self) -> bool:
        """Return whether effective endpoint appears to be Azure OpenAI."""

        base_url = self.resolved_openai_base_url()
        return "azure.com" in base_url.lower()

    def resolved_openai_api_key(self) -> str:
        """Resolve API key with Azure-aware safeguard."""

        if self.uses_azure_openai():
            return self.azure_openai_api_key.strip()
        if self.openai_api_key.strip():
            return self.openai_api_key.strip()
        return self.azure_openai_api_key.strip()

    def resolved_openai_model(self) -> str:
        """Resolve model/deployment name with Azure-aware safeguard."""

        if self.uses_azure_openai() and self.azure_openai_deployment.strip():
            return self.azure_openai_deployment.strip()
        return self.openai_model.strip()

    def resolved_openai_key_source(self) -> str:
        """Return non-secret key source label for diagnostics."""

        if self.uses_azure_openai():
            if self.azure_openai_api_key.strip():
                return "AZURE_OPENAI_API_KEY"
            return "AZURE_OPENAI_API_KEY (missing)"
        if self.openai_api_key.strip():
            return "OPENAI_API_KEY"
        if self.azure_openai_api_key.strip():
            return "AZURE_OPENAI_API_KEY"
        return "none"
