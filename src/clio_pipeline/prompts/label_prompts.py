"""Prompts for Phase 4 cluster labeling and hierarchy scaffolding."""

from __future__ import annotations

CLUSTER_LABEL_SYSTEM_PROMPT = """You are a privacy-preserving clustering analyst.
You receive conversation summary samples for one semantic cluster.

Return strict JSON with exactly these keys:
{
  "name": "<short action-oriented cluster name, max 10 words>",
  "description": "<one or two concise sentences in past tense>"
}

Requirements:
- Be specific and useful for observability.
- Assume neither good nor bad faith.
- If content is harmful, describe it clearly at high level.
- Do not include proper nouns or personally identifying details.
"""

HIERARCHY_LABEL_SYSTEM_PROMPT = """You are organizing cluster labels into higher-level themes.
You receive a list of lower-level cluster names and descriptions.

Return strict JSON with exactly these keys:
{
  "name": "<short higher-level category name, max 10 words>",
  "description": "<one or two concise sentences describing the group>"
}

Requirements:
- Be specific but broader than child clusters.
- Preserve observability and safety signal clarity.
- Do not include proper nouns or personally identifying details.
"""


def build_cluster_label_user_prompt(
    *,
    cluster_id: int,
    size: int,
    unique_users: int,
    summaries: list[str],
) -> str:
    """Build user prompt for naming a base cluster."""

    lines = [f"- {summary}" for summary in summaries]
    samples_text = "\n".join(lines)
    return (
        "Create a name and description for this semantic cluster.\n"
        f"cluster_id: {cluster_id}\n"
        f"cluster_size: {size}\n"
        f"cluster_unique_users: {unique_users}\n"
        "Sample conversation summaries:\n"
        f"{samples_text}\n"
    )


def build_hierarchy_label_user_prompt(
    *,
    group_index: int,
    child_clusters: list[dict],
) -> str:
    """Build user prompt for naming a higher-level parent cluster."""

    lines: list[str] = []
    for child in child_clusters:
        lines.append(
            f"- child_cluster_id={child['cluster_id']}; "
            f"name={child['name']}; description={child['description']}"
        )

    children_text = "\n".join(lines)
    return (
        "Create a higher-level category for the following child clusters.\n"
        f"group_index: {group_index}\n"
        "Child clusters:\n"
        f"{children_text}\n"
    )
