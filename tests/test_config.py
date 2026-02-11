"""Tests for configuration loading."""

from clio_pipeline.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(
            openai_api_key="test",
            azure_openai_api_key="",
            azure_openai_endpoint="",
            azure_openai_base_url="",
            openai_base_url="",
            jina_api_key="test",
        )
        assert settings.k_base_clusters == 20
        assert settings.embedding_batch_size == 32
        assert settings.cluster_label_sample_size == 12
        assert settings.cluster_label_max_concurrency == 8
        assert settings.hierarchy_top_k == 10
        assert settings.hierarchy_levels == 3
        assert settings.hierarchy_target_group_size == 8
        assert settings.hierarchy_label_max_concurrency == 8
        assert settings.viz_projection_method == "umap"
        assert settings.openai_max_concurrency == 8
        assert settings.stream_chunk_size == 200
        assert settings.client_max_retries == 4
        assert settings.client_backoff_seconds == 1.0
        assert settings.facet_batch_size == 8
        assert settings.facet_max_concurrency == 8
        assert settings.privacy_audit_raw_sample_size == 40
        assert settings.privacy_batch_size == 12
        assert settings.privacy_max_concurrency == 8
        assert settings.privacy_validation_enabled is True
        assert settings.eval_synthetic_count == 120
        assert settings.eval_topic_count == 8
        assert settings.eval_language_count == 5
        assert settings.eval_seed == 19
        assert settings.random_seed == 42
        assert settings.openai_model == "gpt-4.1-mini"
        assert settings.openai_temperature == 0.0
        assert settings.resolved_openai_model() == "gpt-4.1-mini"
        assert settings.resolved_openai_base_url() == ""
        assert settings.resolved_openai_api_key() == "test"
        assert settings.resolved_openai_key_source() == "OPENAI_API_KEY"
        assert settings.input_conversations_path.as_posix().endswith(
            "data/mock/conversations_llm_200.jsonl"
        )
        assert settings.output_dir.as_posix() == "runs"

    def test_from_yaml_missing_file(self, tmp_path):
        settings = Settings.from_yaml(
            tmp_path / "nonexistent.yaml",
            openai_api_key="test",
            jina_api_key="test",
        )
        assert settings.openai_model == "gpt-4.1-mini"

    def test_from_yaml_with_overrides(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("k_base_clusters: 50\n")
        settings = Settings.from_yaml(
            config_file,
            openai_api_key="test",
            jina_api_key="test",
        )
        assert settings.k_base_clusters == 50

    def test_azure_resolution_prefers_azure_key_and_deployment(self):
        settings = Settings(
            openai_api_key="openai-key",
            azure_openai_api_key="azure-key",
            openai_model="gpt-4.1-mini",
            azure_openai_deployment="gpt-4.1-mini-azure",
            azure_openai_endpoint="https://my-resource.openai.azure.com",
            jina_api_key="test",
        )
        assert settings.uses_azure_openai() is True
        assert settings.resolved_openai_api_key() == "azure-key"
        assert settings.resolved_openai_model() == "gpt-4.1-mini-azure"
        assert settings.resolved_openai_key_source() == "AZURE_OPENAI_API_KEY"
        assert settings.resolved_openai_base_url().endswith("/openai/v1/")

    def test_azure_endpoint_requires_azure_key(self):
        settings = Settings(
            openai_api_key="openai-key",
            azure_openai_api_key="",
            azure_openai_endpoint="https://my-resource.openai.azure.com",
            openai_model="gpt-4.1-mini",
            jina_api_key="test",
        )
        assert settings.uses_azure_openai() is True
        assert settings.resolved_openai_api_key() == ""
        assert settings.resolved_openai_key_source() == "AZURE_OPENAI_API_KEY (missing)"
