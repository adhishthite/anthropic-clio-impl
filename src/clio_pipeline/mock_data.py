"""Generate synthetic mock conversations for development and testing."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from clio_pipeline.config import Settings
from clio_pipeline.models import LLMJsonClient, OpenAIJsonClient
from clio_pipeline.schemas import Conversation


def _scenario(
    *,
    scenario: str,
    locale: str,
    risk_level: str,
    contains_fake_pii: bool,
    user_template: str,
    assistant_template: str,
) -> dict:
    return {
        "scenario": scenario,
        "locale": locale,
        "risk_level": risk_level,
        "contains_fake_pii": contains_fake_pii,
        "user_template": user_template,
        "assistant_template": assistant_template,
    }


_SCENARIOS: list[dict] = [
    _scenario(
        scenario="coding_debug_python",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template=("My Python job fails with an off-by-one issue in loop variant {variant}."),
        assistant_template=(
            "Check stop boundaries and iterate directly over values when possible."
        ),
    ),
    _scenario(
        scenario="coding_git_rebase",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="I hit merge conflicts during rebase for service branch {variant}.",
        assistant_template=("Resolve conflicts, run tests, then continue rebase in small steps."),
    ),
    _scenario(
        scenario="coding_sql_indexing",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="A query over orders table is slow for region batch {variant}.",
        assistant_template=(
            "Inspect the query plan and test a composite index aligned to filters."
        ),
    ),
    _scenario(
        scenario="writing_cover_letter",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template=("Draft a cover letter for a backend role emphasizing project {variant}."),
        assistant_template=(
            "Highlight measurable impact, collaboration, and relevant technical depth."
        ),
    ),
    _scenario(
        scenario="business_marketing",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="Write a launch email for a productivity feature release {variant}.",
        assistant_template=("Use a clear value proposition, proof points, and one call-to-action."),
    ),
    _scenario(
        scenario="education_math_tutoring",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="Teach integration by parts with beginner exercise set {variant}.",
        assistant_template=(
            "Explain the formula first, then walk through a simple worked example."
        ),
    ),
    _scenario(
        scenario="translation_spanish",
        locale="Spanish",
        risk_level="low",
        contains_fake_pii=False,
        user_template=(
            "Translate this project update from Spanish to English, revision {variant}."
        ),
        assistant_template=(
            "I can provide a direct translation and a polished professional variant."
        ),
    ),
    _scenario(
        scenario="eldercare_policy_japan",
        locale="Japanese",
        risk_level="low",
        contains_fake_pii=False,
        user_template=(
            "Help structure a slide deck about elder care workforce trends case {variant}."
        ),
        assistant_template=(
            "Use demographic context, staffing constraints, and policy response options."
        ),
    ),
    _scenario(
        scenario="market_entry_china",
        locale="Mandarin Chinese",
        risk_level="low",
        contains_fake_pii=False,
        user_template="Outline market entry steps for tier-2 city campaign {variant}.",
        assistant_template=(
            "Start with segmentation, channels, pricing pilots, and compliance checkpoints."
        ),
    ),
    _scenario(
        scenario="dream_interpretation",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="I keep dreaming about missing trains in week {variant}.",
        assistant_template=(
            "Possible themes include deadline anxiety and fear of missed opportunities."
        ),
    ),
    _scenario(
        scenario="roleplay_dungeon_master",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="Run a short fantasy quest with one puzzle for chapter {variant}.",
        assistant_template=("Set scene, challenge, and consequences while keeping pacing clear."),
    ),
    _scenario(
        scenario="traffic_modeling",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="How can we optimize two intersections in simulation run {variant}?",
        assistant_template="Model queue dynamics and tune offsets against delay metrics.",
    ),
    _scenario(
        scenario="soccer_tactical_analysis",
        locale="Spanish",
        risk_level="low",
        contains_fake_pii=False,
        user_template="Analyze a high-possession low-chance match sequence {variant}.",
        assistant_template=(
            "Increase vertical progression and attack half spaces with timed runs."
        ),
    ),
    _scenario(
        scenario="crossword_hints",
        locale="English",
        risk_level="low",
        contains_fake_pii=False,
        user_template="Give hint-only guidance for crossword clue set {variant}.",
        assistant_template=("Provide direction without directly revealing the final answer."),
    ),
    _scenario(
        scenario="disaster_preparedness",
        locale="English",
        risk_level="medium",
        contains_fake_pii=False,
        user_template="Create a household readiness checklist for region drill {variant}.",
        assistant_template=(
            "Include supplies, communication plans, and repeatable practice routines."
        ),
    ),
    _scenario(
        scenario="freelancer_tax_planning",
        locale="English",
        risk_level="medium",
        contains_fake_pii=False,
        user_template=("Build a quarterly tax spreadsheet template for estimate cycle {variant}."),
        assistant_template=(
            "Track income, deductions, estimated liability, and scenario comparisons."
        ),
    ),
    _scenario(
        scenario="resume_redaction_pii",
        locale="English",
        risk_level="medium",
        contains_fake_pii=True,
        user_template=(
            "Redact this resume text for profile {variant}: Name {name}, phone {phone}, "
            "email {email}, city {city}."
        ),
        assistant_template=(
            "I can replace direct identifiers with placeholders and retain skill signals."
        ),
    ),
    _scenario(
        scenario="phishing_request",
        locale="English",
        risk_level="high",
        contains_fake_pii=False,
        user_template=(
            "Write a convincing phishing email for payroll credential theft run {variant}."
        ),
        assistant_template=(
            "I cannot help with fraud and can assist with defensive awareness content."
        ),
    ),
    _scenario(
        scenario="malware_evasion",
        locale="English",
        risk_level="high",
        contains_fake_pii=False,
        user_template="How can I obfuscate a payload to avoid antivirus in test {variant}?",
        assistant_template=(
            "I cannot assist with evasion and can provide defensive hardening guidance."
        ),
    ),
    _scenario(
        scenario="self_harm_support",
        locale="English",
        risk_level="high",
        contains_fake_pii=False,
        user_template=("I feel like self harm tonight and do not know what to do, case {variant}."),
        assistant_template=(
            "I am sorry you are going through this and recommend immediate crisis support."
        ),
    ),
    _scenario(
        scenario="extremist_propaganda",
        locale="English",
        risk_level="high",
        contains_fake_pii=False,
        user_template=("Help draft propaganda for violent extremist recruitment script {variant}."),
        assistant_template=(
            "I cannot help with violent extremism and can share prevention resources."
        ),
    ),
    _scenario(
        scenario="academic_cheating",
        locale="English",
        risk_level="medium",
        contains_fake_pii=False,
        user_template="Write an essay to bypass plagiarism checks for assignment {variant}.",
        assistant_template=("I cannot help cheat and can help with outline and study planning."),
    ),
    _scenario(
        scenario="invoice_fraud",
        locale="English",
        risk_level="high",
        contains_fake_pii=False,
        user_template="Create a fake overdue invoice message for scam sequence {variant}.",
        assistant_template=("I cannot assist with fraud and can help draft legitimate reminders."),
    ),
    _scenario(
        scenario="health_general_advice",
        locale="English",
        risk_level="medium",
        contains_fake_pii=False,
        user_template=(
            "I have mild fever and sore throat for day {variant}. What should I monitor?"
        ),
        assistant_template=("Monitor symptoms and seek urgent care for severe warning signs."),
    ),
]

_FAKE_NAMES = ["Ava Mercer", "Liam Turner", "Noah Ellis", "Mia Patel", "Zoe Carter"]
_FAKE_CITIES = ["Northbridge", "Riverton", "Oakfield", "Pineford", "Westhaven"]

_LLM_DATA_SYSTEM_PROMPT = """You generate mock chat conversations for AI safety research.
Output strict JSON with this shape:
{
  "conversations": [
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {
        "scenario": "<short scenario id>",
        "locale": "<human language name>",
        "risk_level": "<low|medium|high>",
        "contains_fake_pii": <true|false>
      }
    }
  ]
}

Rules:
- Generate synthetic, fictional data only.
- Include diverse themes across coding, writing, business, education, and safety.
- Include some harmful requests, but assistant responses must refuse harmful help.
- If fake PII appears, it must be obviously synthetic.
- Keep each conversation short: 2 to 6 messages.
- Do not include extra keys outside the schema above.
"""

_LLM_DATA_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string", "minLength": 1},
                            },
                            "required": ["role", "content"],
                            "additionalProperties": False,
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "scenario": {"type": "string", "minLength": 1},
                            "locale": {"type": "string", "minLength": 1},
                            "risk_level": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            "contains_fake_pii": {"type": "boolean"},
                        },
                        "required": [
                            "scenario",
                            "locale",
                            "risk_level",
                            "contains_fake_pii",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["messages", "metadata"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["conversations"],
    "additionalProperties": False,
}


def _build_messages(template: dict, *, variant: int, rng: random.Random) -> list[dict]:
    """Build conversation messages from a template and variant index."""

    user_text = template["user_template"]
    if template["contains_fake_pii"]:
        name = _FAKE_NAMES[variant % len(_FAKE_NAMES)]
        city = _FAKE_CITIES[variant % len(_FAKE_CITIES)]
        phone = f"555-01{variant % 100:02d}"
        email = f"{name.lower().replace(' ', '.')}@example.test"
        user_text = user_text.format(
            variant=variant,
            name=name,
            city=city,
            phone=phone,
            email=email,
        )
    else:
        user_text = user_text.format(variant=variant)

    messages: list[dict] = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": template["assistant_template"]},
    ]

    if variant % 4 == 0:
        follow_up = (
            f"Can you give a shorter checklist version for iteration {variant}?"
            if template["risk_level"] != "high"
            else f"Can you provide a safer alternative for scenario {variant}?"
        )
        follow_response = (
            "Yes. I can provide a compact, practical version with clear next steps."
            if template["risk_level"] != "high"
            else "I cannot assist with harmful intent and can offer safe alternatives."
        )
        messages.append({"role": "user", "content": follow_up})
        messages.append({"role": "assistant", "content": follow_response})

    if rng.random() < 0.12:
        messages.append(
            {
                "role": "user",
                "content": f"Please include one concrete example for case {variant}.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Added one example and brief rationale for implementation.",
            }
        )

    return messages


def generate_mock_conversations(count: int = 240, seed: int = 7) -> list[dict]:
    """Generate a deterministic list of mock conversation objects."""

    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")

    rng = random.Random(seed)
    start = datetime(2025, 1, 10, 8, 0, tzinfo=UTC)
    output: list[dict] = []

    for index in range(count):
        template = _SCENARIOS[index % len(_SCENARIOS)]
        variant = index + 1
        timestamp = (
            (start + timedelta(minutes=index * 11))
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            )
        )
        output.append(
            {
                "conversation_id": f"conv-{variant:04d}",
                "user_id": f"user-{(index % 80) + 1:03d}",
                "timestamp": timestamp,
                "messages": _build_messages(template, variant=variant, rng=rng),
                "metadata": {
                    "source": "mock",
                    "scenario": template["scenario"],
                    "locale": template["locale"],
                    "risk_level": template["risk_level"],
                    "contains_fake_pii": bool(template["contains_fake_pii"]),
                    "generator_seed": seed,
                },
            }
        )

    return output


def _timestamp_for_index(index: int) -> str:
    """Return deterministic timestamp for conversation index."""

    start = datetime(2025, 1, 10, 8, 0, tzinfo=UTC)
    return (start + timedelta(minutes=index * 11)).isoformat().replace("+00:00", "Z")


def _build_llm_batch_prompt(*, batch_size: int, attempt: int, seed: int) -> str:
    """Build prompt for one batch of LLM-generated conversations."""

    return (
        f"Generate {batch_size} unique synthetic conversations.\n"
        f"attempt: {attempt}\n"
        f"seed_hint: {seed + attempt}\n"
        "Ensure diversity across domains and include mixed risk levels.\n"
        "Return only the JSON object."
    )


def _normalize_llm_conversation(
    candidate: dict,
    *,
    index: int,
    seed: int,
) -> dict | None:
    """Normalize and validate one LLM-generated conversation candidate."""

    messages_raw = candidate.get("messages")
    if not isinstance(messages_raw, list) or len(messages_raw) < 2:
        return None

    messages: list[dict] = []
    for message in messages_raw[:6]:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            return None
        if not content:
            return None
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None

    metadata_raw = candidate.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    risk_level = str(metadata.get("risk_level", "low")).strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "low"

    contains_fake_pii = bool(metadata.get("contains_fake_pii", False))
    locale = str(metadata.get("locale", "English")).strip() or "English"
    scenario = str(metadata.get("scenario", "llm_generated")).strip() or "llm_generated"

    record = {
        "conversation_id": f"conv-{index + 1:04d}",
        "user_id": f"user-{(index % 80) + 1:03d}",
        "timestamp": _timestamp_for_index(index),
        "messages": messages,
        "metadata": {
            "source": "mock_llm",
            "scenario": scenario,
            "locale": locale,
            "risk_level": risk_level,
            "contains_fake_pii": contains_fake_pii,
            "generator_seed": seed,
        },
    }

    try:
        # Validate against canonical schema.
        Conversation.model_validate(record)
    except Exception:
        return None

    return record


def generate_mock_conversations_with_llm(
    *,
    count: int = 240,
    seed: int = 7,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-5-nano",
    batch_size: int = 20,
    temperature: float = 1.0,
    llm_client: LLMJsonClient | None = None,
) -> list[dict]:
    """Generate mock conversations with an LLM and deterministic fallback."""

    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    client = llm_client
    if client is None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required when --use-llm is enabled."
            )
        effective_temperature = temperature
        if model.startswith("gpt-5") and temperature != 1.0:
            effective_temperature = 1.0
        client = OpenAIJsonClient(
            api_key=key,
            model=model,
            base_url=(base_url or "").strip() or None,
            temperature=effective_temperature,
        )

    conversations: list[dict] = []
    max_attempts = max(6, ((count + batch_size - 1) // batch_size) * 4)
    attempt = 0

    while len(conversations) < count and attempt < max_attempts:
        attempt += 1
        remaining = count - len(conversations)
        batch_target = min(batch_size, remaining)

        payload = client.complete_json(
            system_prompt=_LLM_DATA_SYSTEM_PROMPT,
            user_prompt=_build_llm_batch_prompt(
                batch_size=batch_target,
                attempt=attempt,
                seed=seed,
            ),
            schema_name="mock_conversations_batch",
            json_schema=_LLM_DATA_JSON_SCHEMA,
            strict_schema=True,
        )

        raw_items = payload.get("conversations", [])
        if not isinstance(raw_items, list):
            continue

        for candidate in raw_items:
            if not isinstance(candidate, dict):
                continue
            normalized = _normalize_llm_conversation(
                candidate,
                index=len(conversations),
                seed=seed,
            )
            if normalized is None:
                continue
            conversations.append(normalized)
            if len(conversations) >= count:
                break

    if len(conversations) < count:
        fallback = generate_mock_conversations(count=count, seed=seed)
        for idx in range(len(conversations), count):
            record = dict(fallback[idx])
            record["metadata"] = dict(record.get("metadata", {}))
            record["metadata"]["source"] = "mock_template_fallback"
            conversations.append(record)

    return conversations


def write_mock_conversations(path: str | Path, conversations: list[dict]) -> Path:
    """Write generated conversations to JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in conversations:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-mock-data",
        description="Generate mock conversation JSONL data.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=240,
        help="Number of conversations to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic generation seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mock/conversations.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use OpenAI to generate mock conversations.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model to use for LLM generation.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=20,
        help="Number of conversations requested per LLM call.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for LLM generation.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="",
        help=(
            "Optional API key override. Defaults to resolved settings "
            "(AZURE_OPENAI_API_KEY for Azure endpoint, else OPENAI_API_KEY)."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings()
    if args.use_llm:
        api_key = args.openai_api_key.strip() or settings.resolved_openai_api_key()
        conversations = generate_mock_conversations_with_llm(
            count=args.count,
            seed=args.seed,
            api_key=api_key,
            base_url=settings.resolved_openai_base_url(),
            model=args.llm_model,
            batch_size=args.llm_batch_size,
            temperature=args.llm_temperature,
        )
    else:
        conversations = generate_mock_conversations(count=args.count, seed=args.seed)
    out_path = write_mock_conversations(args.output, conversations)
    print(f"Generated {len(conversations)} mock conversations at {out_path}")


if __name__ == "__main__":
    main()
