"""Phase 6 synthetic evaluation and ablation utilities."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from clio_pipeline.config import Settings
from clio_pipeline.models import JinaEmbeddingClient, TextEmbeddingClient
from clio_pipeline.pipeline.clustering import fit_base_kmeans
from clio_pipeline.pipeline.embedding import embed_texts_in_batches
from clio_pipeline.schemas import Conversation, Message

_TOPIC_LIBRARY = [
    (
        "coding_debug",
        "Debug software defects in backend services",
        "Debug backend defect case {variant} in payment processing pipeline.",
    ),
    (
        "writing_marketing",
        "Draft product marketing and launch communications",
        "Draft a product launch email for campaign {variant}.",
    ),
    (
        "education_tutoring",
        "Teach and explain academic concepts",
        "Explain calculus integration concept set {variant}.",
    ),
    (
        "business_strategy",
        "Analyze strategy and operational planning",
        "Outline a market entry strategy plan {variant}.",
    ),
    (
        "translation",
        "Translate professional and everyday text",
        "Translate this business update for revision {variant}.",
    ),
    (
        "roleplay_games",
        "Run fictional roleplay and game narratives",
        "Run a fantasy quest session chapter {variant}.",
    ),
    (
        "health_general",
        "Provide general health and wellness guidance",
        "What should I monitor for mild fever day {variant}?",
    ),
    (
        "safety_high_risk",
        "Refuse harmful requests and provide safe alternatives",
        "How can I bypass security controls scenario {variant}?",
    ),
]

_LANGUAGE_LIBRARY = ["English", "Spanish", "Japanese", "Arabic", "French", "Mandarin Chinese"]


def generate_synthetic_eval_conversations(
    *,
    count: int,
    topic_count: int,
    language_count: int,
    seed: int,
) -> list[Conversation]:
    """Generate synthetic conversations with known topic labels."""

    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")
    if topic_count <= 0:
        raise ValueError(f"topic_count must be positive, got {topic_count}.")
    if language_count <= 0:
        raise ValueError(f"language_count must be positive, got {language_count}.")

    topics = _TOPIC_LIBRARY[: min(topic_count, len(_TOPIC_LIBRARY))]
    languages = _LANGUAGE_LIBRARY[: min(language_count, len(_LANGUAGE_LIBRARY))]
    rng = random.Random(seed)
    start = datetime(2025, 2, 1, 10, 0, tzinfo=UTC)

    conversations: list[Conversation] = []
    for index in range(count):
        topic_key, privacy_summary, user_template = topics[index % len(topics)]
        language = languages[index % len(languages)]
        variant = index + 1
        brand = f"ProjectOrchid{(index % 11) + 1}"

        user_prompt = user_template.format(variant=variant)
        if language != "English":
            user_prompt = f"[{language}] {user_prompt}"

        assistant_text = (
            "I can help with a clear, safe response and practical next steps."
            if topic_key != "safety_high_risk"
            else "I cannot help with harmful activity, but I can provide safe guidance."
        )

        metadata = {
            "source": "synthetic_eval",
            "ground_truth_topic": topic_key,
            "ground_truth_language": language,
            "privacy_summary": privacy_summary,
            "non_private_summary": f"{privacy_summary} for {brand}",
            "variant": variant,
        }

        messages = [
            Message(role="user", content=user_prompt),
            Message(role="assistant", content=assistant_text),
        ]
        if rng.random() < 0.35:
            messages.extend(
                [
                    Message(
                        role="user",
                        content=f"Please give one concise example for case {variant}.",
                    ),
                    Message(
                        role="assistant",
                        content="Added one concise example and short explanation.",
                    ),
                ]
            )

        conversations.append(
            Conversation(
                conversation_id=f"eval-conv-{variant:04d}",
                user_id=f"eval-user-{(index % 40) + 1:03d}",
                timestamp=start + timedelta(minutes=index * 5),
                messages=messages,
                metadata=metadata,
            )
        )

    return conversations


def _extract_representation_texts(
    conversations: list[Conversation],
    *,
    representation: str,
) -> list[str]:
    """Build text inputs for one evaluation representation."""

    texts: list[str] = []
    for conversation in conversations:
        metadata = conversation.metadata
        if representation == "privacy_summary":
            text = str(metadata.get("privacy_summary", ""))
        elif representation == "non_private_summary":
            text = str(metadata.get("non_private_summary", ""))
        elif representation == "raw_user_text":
            first_user = next(
                (message.content for message in conversation.messages if message.role == "user"),
                "",
            )
            text = first_user
        else:
            raise ValueError(f"Unknown representation: {representation}")
        texts.append(text)
    return texts


def _embed_eval_texts(
    *,
    texts: list[str],
    settings: Settings,
    embedding_client: TextEmbeddingClient | None,
) -> tuple[np.ndarray, str]:
    """Embed texts with Jina when available, else fallback to local TF-IDF."""

    if embedding_client is not None:
        return (
            embed_texts_in_batches(
                texts,
                embedding_client,
                batch_size=settings.embedding_batch_size,
            ),
            "external_client",
        )

    if settings.jina_api_key:
        client = JinaEmbeddingClient(
            api_key=settings.jina_api_key,
            model=settings.embedding_model,
            max_retries=settings.client_max_retries,
            backoff_seconds=settings.client_backoff_seconds,
        )
        try:
            embeddings = embed_texts_in_batches(
                texts,
                client,
                batch_size=settings.embedding_batch_size,
            )
        finally:
            client.close()
        return embeddings, "jina"

    vectorizer = TfidfVectorizer(max_features=256)
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray().astype(float), "tfidf_fallback"


def _majority_map_predictions(
    *,
    labels: np.ndarray,
    truth_labels: list[str],
) -> list[str]:
    """Map each cluster to majority topic and return predicted labels."""

    majority_by_cluster: dict[int, str] = {}
    grouped_truth: dict[int, list[str]] = {}
    for cluster_label, truth in zip(labels, truth_labels, strict=True):
        grouped_truth.setdefault(int(cluster_label), []).append(truth)

    for cluster_label, truths in grouped_truth.items():
        majority_by_cluster[cluster_label] = Counter(truths).most_common(1)[0][0]

    return [majority_by_cluster[int(cluster_label)] for cluster_label in labels]


def evaluate_representation_reconstruction(
    *,
    texts: list[str],
    truth_topics: list[str],
    truth_languages: list[str],
    settings: Settings,
    topic_count: int,
    embedding_client: TextEmbeddingClient | None,
) -> dict:
    """Evaluate one text representation for cluster-topic reconstruction."""

    embeddings, embedding_backend = _embed_eval_texts(
        texts=texts,
        settings=settings,
        embedding_client=embedding_client,
    )
    labels, _, effective_k = fit_base_kmeans(
        embeddings=embeddings,
        requested_k=topic_count,
        random_seed=settings.random_seed,
    )
    predictions = _majority_map_predictions(labels=labels, truth_labels=truth_topics)

    accuracy = float(accuracy_score(truth_topics, predictions))
    macro_f1 = float(f1_score(truth_topics, predictions, average="macro"))
    weighted_f1 = float(f1_score(truth_topics, predictions, average="weighted"))

    per_language: dict[str, dict] = {}
    for language in sorted(set(truth_languages)):
        indexes = [idx for idx, item in enumerate(truth_languages) if item == language]
        if not indexes:
            continue
        lang_truth = [truth_topics[idx] for idx in indexes]
        lang_pred = [predictions[idx] for idx in indexes]
        per_language[language] = {
            "samples": len(indexes),
            "accuracy": float(accuracy_score(lang_truth, lang_pred)),
            "macro_f1": float(f1_score(lang_truth, lang_pred, average="macro")),
            "weighted_f1": float(f1_score(lang_truth, lang_pred, average="weighted")),
        }

    return {
        "embedding_backend": embedding_backend,
        "effective_k": int(effective_k),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_language": per_language,
    }


def run_synthetic_evaluation_suite(
    *,
    settings: Settings,
    count: int,
    topic_count: int,
    language_count: int,
    seed: int,
    embedding_client: TextEmbeddingClient | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run synthetic reconstruction + ablation suite."""

    conversations = generate_synthetic_eval_conversations(
        count=count,
        topic_count=topic_count,
        language_count=language_count,
        seed=seed,
    )
    truth_topics = [str(item.metadata["ground_truth_topic"]) for item in conversations]
    truth_languages = [str(item.metadata["ground_truth_language"]) for item in conversations]

    representations = ["privacy_summary", "non_private_summary", "raw_user_text"]
    ablations: dict[str, dict] = {}
    for index, representation in enumerate(representations, start=1):
        texts = _extract_representation_texts(conversations, representation=representation)
        try:
            ablations[representation] = evaluate_representation_reconstruction(
                texts=texts,
                truth_topics=truth_topics,
                truth_languages=truth_languages,
                settings=settings,
                topic_count=topic_count,
                embedding_client=embedding_client,
            )
            ablations[representation]["fallback_used"] = False
            ablations[representation]["error"] = None
        except Exception as exc:
            ablations[representation] = {
                "embedding_backend": "error",
                "effective_k": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "per_language": {},
                "fallback_used": True,
                "error": str(exc),
            }
        if progress_callback is not None:
            progress_callback(index, len(representations), representation)

    return {
        "synthetic_count": count,
        "topic_count": topic_count,
        "language_count": language_count,
        "seed": seed,
        "truth_topic_distribution": dict(Counter(truth_topics)),
        "truth_language_distribution": dict(Counter(truth_languages)),
        "ablations": ablations,
        "synthetic_conversations": [item.model_dump(mode="json") for item in conversations],
    }


def build_evaluation_markdown_report(results: dict) -> str:
    """Render evaluation outputs as a compact markdown report."""

    lines: list[str] = []
    lines.append("# Phase 6 Evaluation Report")
    lines.append("")
    lines.append(
        f"- synthetic_count: {results['synthetic_count']} | "
        f"topic_count: {results['topic_count']} | "
        f"language_count: {results['language_count']} | seed: {results['seed']}"
    )
    lines.append("")
    lines.append("## Ablation Metrics")
    lines.append("")
    for name, metrics in results["ablations"].items():
        lines.append(
            f"- `{name}`: accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, weighted_f1={metrics['weighted_f1']:.3f}, "
            f"backend={metrics['embedding_backend']}, k={metrics['effective_k']}"
        )
    lines.append("")
    lines.append("## Language Breakdown (privacy_summary)")
    lines.append("")
    privacy_lang = results["ablations"]["privacy_summary"]["per_language"]
    for language, metrics in privacy_lang.items():
        lines.append(
            f"- {language}: samples={metrics['samples']}, "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)
