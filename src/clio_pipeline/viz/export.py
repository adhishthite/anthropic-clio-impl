"""Projection and tree export utilities."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def project_embeddings_2d(
    embeddings: np.ndarray,
    *,
    method: str,
    random_seed: int,
) -> tuple[np.ndarray, str]:
    """Project high-dimensional embeddings to 2D for map views."""

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got ndim={embeddings.ndim}.")
    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings cannot be empty for projection.")

    method_lower = method.lower()
    if method_lower == "umap":
        try:
            import umap.umap_ as umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.0,
                metric="cosine",
                random_state=random_seed,
            )
            return reducer.fit_transform(embeddings), "umap"
        except Exception:
            pass

    pca = PCA(n_components=2, random_state=random_seed)
    return pca.fit_transform(embeddings), "pca"


def build_map_points(
    *,
    facets: list,
    assignments: list[dict],
    coords: np.ndarray,
) -> list[dict]:
    """Build per-conversation map points from projected embeddings."""

    assignment_by_id = {item["conversation_id"]: item for item in assignments}
    points: list[dict] = []
    for idx, facet in enumerate(facets):
        assignment = assignment_by_id.get(facet.conversation_id, {})
        points.append(
            {
                "conversation_id": facet.conversation_id,
                "x": float(coords[idx][0]),
                "y": float(coords[idx][1]),
                "cluster_id": int(assignment.get("cluster_id", -1)),
                "kept_by_threshold": bool(assignment.get("kept_by_threshold", False)),
                "language": facet.language,
                "concerning_score": int(facet.concerning_score),
            }
        )
    return points


def build_cluster_map(points: list[dict]) -> list[dict]:
    """Build cluster-level map centroids from point data."""

    grouped: dict[int, list[dict]] = {}
    for point in points:
        grouped.setdefault(int(point["cluster_id"]), []).append(point)

    clusters: list[dict] = []
    for cluster_id in sorted(grouped.keys()):
        rows = grouped[cluster_id]
        x = float(sum(item["x"] for item in rows) / len(rows))
        y = float(sum(item["y"] for item in rows) / len(rows))
        clusters.append(
            {
                "cluster_id": cluster_id,
                "x": x,
                "y": y,
                "size": len(rows),
                "kept_by_threshold": bool(rows[0]["kept_by_threshold"]),
                "avg_concerning_score": float(
                    sum(item["concerning_score"] for item in rows) / len(rows)
                ),
            }
        )
    return clusters


def hierarchy_to_tree_view(hierarchy: dict) -> dict:
    """Convert hierarchy output to a UI-friendly tree payload."""

    nodes = hierarchy.get("nodes", [])
    edges = hierarchy.get("edges", [])

    by_id = {str(node["node_id"]): dict(node) for node in nodes if "node_id" in node}
    children_by_parent: dict[str, list[str]] = {}
    for edge in edges:
        parent = str(edge["parent_id"])
        child = str(edge["child_id"])
        children_by_parent.setdefault(parent, []).append(child)

    for node_id, node in by_id.items():
        node["children"] = children_by_parent.get(node_id, [])

    roots = [
        node
        for node in by_id.values()
        if int(node.get("level", 0)) == hierarchy.get("max_level", 0)
    ]
    return {
        "max_level": hierarchy.get("max_level", 0),
        "roots": roots,
        "node_count": len(by_id),
        "edge_count": len(edges),
        "nodes": list(by_id.values()),
        "edges": edges,
    }
