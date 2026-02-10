# Input JSONL Contract (v1)

This document defines the canonical input format expected by `ant-clio` when loading
conversation data from any upstream store (MongoDB, Elasticsearch, Postgres, files, etc.).

## Required File Format

- File type: JSONL (one JSON object per line)
- Encoding: UTF-8
- Each non-empty line must represent exactly one conversation record
- `conversation_id` values must be unique across the file

## Required Object Shape

Each line must match this object structure:

```json
{
  "conversation_id": "conv_123",
  "user_id": "user_456",
  "timestamp": "2026-02-10T12:34:56Z",
  "messages": [
    { "role": "user", "content": "How do I cancel my subscription?" },
    {
      "role": "assistant",
      "content": "I can walk you through cancellation steps."
    }
  ],
  "metadata": {
    "source": "postgres",
    "tenant": "acme",
    "channel": "webchat"
  }
}
```

### Field Definitions

- `conversation_id` (string, required): unique conversation identifier
- `user_id` (string, required): stable user identifier for privacy thresholding
- `timestamp` (ISO-8601 datetime string, required): conversation timestamp
- `messages` (array, required): ordered message list
  - `role` (string, required): message role (`user` / `assistant` recommended)
  - `content` (string, required): message text content
- `metadata` (object, optional but recommended): free-form source context

## Canonical JSON Schema (reference)

```json
{
  "type": "object",
  "properties": {
    "conversation_id": { "type": "string" },
    "user_id": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": { "type": "string" },
          "content": { "type": "string" }
        },
        "required": ["role", "content"],
        "additionalProperties": false
      }
    },
    "metadata": { "type": "object" }
  },
  "required": ["conversation_id", "user_id", "timestamp", "messages"],
  "additionalProperties": true
}
```

## Validate Before Running

Run the validator before pipeline execution:

```bash
uv run clio validate-input --input /path/to/conversations.jsonl --report-json /tmp/validation_report.json
```

If validation fails, fix reported lines and rerun validation.

## Mapping Notes for Mongo / Elastic / Postgres

- Map your source's conversation primary key -> `conversation_id`
- Map your source's user identifier -> `user_id`
- Map conversation created time (or first message time) -> `timestamp`
- Normalize messages into one ordered array per conversation:
  - If your source is message-level rows, group by conversation key
  - Sort by sequence number or timestamp before writing `messages`
- Keep source-specific fields in `metadata` to preserve traceability

## Important Privacy Note

The external input file can include `user_id`, `timestamp`, and `metadata` as above.
Inside each run folder, `conversation.jsonl` is intentionally rewritten to messages-only
for privacy-minimized snapshots.
