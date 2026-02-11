# Source Adapter Recipes

This guide shows practical ways to move conversation data from common stores into
the CLIO input contract (`docs/input_jsonl_contract.md`).

## Recommended Pipeline

1. Export raw records from your source into a line-delimited JSON file.
2. Normalize the raw file into canonical CLIO JSONL:
   - `uv run python scripts/normalize_external_jsonl.py --input ... --output ... --rejects ... --report ...`
3. Validate the normalized file:
   - `uv run clio validate-input --input /path/to/normalized.jsonl --report-json /tmp/validation.json`
4. Run CLIO:
   - `uv run clio run --input /path/to/normalized.jsonl --with-hierarchy --with-privacy --with-eval`

## Postgres Export

Example query when messages are stored as one row per message:

```sql
SELECT
  conversation_id,
  user_id,
  MIN(created_at) AS timestamp,
  JSON_AGG(
    JSON_BUILD_OBJECT('role', role, 'content', content)
    ORDER BY created_at ASC
  ) AS messages
FROM chat_messages
GROUP BY conversation_id, user_id;
```

Export query result to JSONL (using your preferred method), then run validator.

## MongoDB Export

If one document already contains one conversation:

```javascript
db.conversations.find(
  {},
  {
    _id: 0,
    conversation_id: 1,
    user_id: 1,
    timestamp: 1,
    messages: 1,
    metadata: 1
  }
)
```

If your schema differs, export raw documents and normalize with
`scripts/normalize_external_jsonl.py`.

## Elasticsearch Export

Use scroll/search_after to export `_source` documents to JSONL. Then normalize:

```bash
uv run python scripts/normalize_external_jsonl.py \
  --input data/external/elastic_raw.jsonl \
  --output data/external/elastic.contract.jsonl \
  --rejects data/external/elastic.rejects.jsonl \
  --report data/external/elastic.report.json
```

## Adapter Design Notes

- Keep canonical fields at top level:
  - `conversation_id`, `user_id`, `timestamp`, `messages`, `metadata`
- Put source-only fields into `metadata` when useful for traceability.
- Preserve message order.
- Normalize roles to `user` / `assistant` when possible.
- Reject or repair empty-message conversations before production runs.
