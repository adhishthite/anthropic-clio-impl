"""Pipeline stage implementations."""

from clio_pipeline.pipeline.clustering import ClusteringError, fit_base_kmeans
from clio_pipeline.pipeline.embedding import EmbeddingExtractionError, embed_texts_in_batches
from clio_pipeline.pipeline.evaluate import (
    build_evaluation_markdown_report,
    run_synthetic_evaluation_suite,
)
from clio_pipeline.pipeline.facet_extraction import (
    FacetExtractionError,
    extract_conversation_facets,
    extract_facets_batch,
)
from clio_pipeline.pipeline.hierarchy import (
    HierarchyError,
    build_hierarchy_scaffold,
    build_multilevel_hierarchy_scaffold,
    fit_hierarchy_groups,
)
from clio_pipeline.pipeline.labeling import ClusterLabelingError, label_clusters
from clio_pipeline.pipeline.privacy_audit import (
    PrivacyAuditError,
    run_privacy_auditor_validation,
)
from clio_pipeline.pipeline.run_pipeline import (
    build_run_fingerprint,
    generate_run_id,
    initialize_run_artifacts,
    initialize_run_artifacts_streaming,
    load_phase2_facets,
    load_phase3_cluster_summaries,
    load_phase4_hierarchy,
    load_phase4_labeled_clusters,
    load_phase5_outputs,
    load_phase6_evaluation,
    run_phase1_dataset_load,
    run_phase1_dataset_load_streaming,
    run_phase2_facet_extraction,
    run_phase2_facet_extraction_streaming,
    run_phase3_base_clustering,
    run_phase4_cluster_labeling,
    run_phase4_hierarchy_scaffold,
    run_phase5_privacy_audit,
    run_phase6_evaluation,
)

__all__ = [
    "ClusterLabelingError",
    "ClusteringError",
    "EmbeddingExtractionError",
    "FacetExtractionError",
    "HierarchyError",
    "PrivacyAuditError",
    "build_evaluation_markdown_report",
    "build_hierarchy_scaffold",
    "build_multilevel_hierarchy_scaffold",
    "build_run_fingerprint",
    "embed_texts_in_batches",
    "extract_conversation_facets",
    "extract_facets_batch",
    "fit_base_kmeans",
    "fit_hierarchy_groups",
    "generate_run_id",
    "initialize_run_artifacts",
    "initialize_run_artifacts_streaming",
    "label_clusters",
    "load_phase2_facets",
    "load_phase3_cluster_summaries",
    "load_phase4_hierarchy",
    "load_phase4_labeled_clusters",
    "load_phase5_outputs",
    "load_phase6_evaluation",
    "run_phase1_dataset_load",
    "run_phase1_dataset_load_streaming",
    "run_phase2_facet_extraction",
    "run_phase2_facet_extraction_streaming",
    "run_phase3_base_clustering",
    "run_phase4_cluster_labeling",
    "run_phase4_hierarchy_scaffold",
    "run_phase5_privacy_audit",
    "run_phase6_evaluation",
    "run_privacy_auditor_validation",
    "run_synthetic_evaluation_suite",
]
