import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

if os.getenv("NEWSLETTER_LOAD_DOTENV") == "1":
    load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0)
NEWSLETTER_INCIDENT_DIR = os.getenv("NEWSLETTER_INCIDENT_DIR", "./incidents")
SCORING_MAX_TOKENS = int(os.getenv("NEWSLETTER_SCORING_MAX_TOKENS", "8000"))
SCORING_ATTEMPTS = int(os.getenv("NEWSLETTER_SCORING_ATTEMPTS", "2"))
VALID_CATEGORIES = {"ai_world", "business_world", "teach_this_candidate", "exclude"}
CATEGORY_ALIASES = {
    "ai": "ai_world",
    "automation": "ai_world",
    "business": "business_world",
    "teach_this": "teach_this_candidate",
    "teaching_candidate": "teach_this_candidate",
    "excluded": "exclude",
}

SCORING_SYSTEM_PROMPT = """\
You score candidate news articles for an SMB operator evaluating practical AI workflows.

Score each article from 0 to 100 for newsletter inclusion.

Scoring criteria:
- SMB/operator relevance: small business, agencies, founders, operations, sales, support, finance, hiring, workflows.
- AI usefulness: AI tools, agents, automations, model updates, integrations, workflows, adoption stories.
- Practicality: clear business use case, implementation detail, cost/pricing, ROI, templates, examples, lessons.
- Teaching value: can be turned into a useful lesson for SMB owners or agency clients.
- Strategic signal: helps Vatsal anticipate client questions, content topics, offers, or tool choices.

Do NOT score based on recency. Every supplied article is already inside the same 30-hour freshness window.
A 2-hour-old article and a 28-hour-old article should score the same if their usefulness is the same.

Penalize:
- Hype, vague predictions, and "AI will replace jobs" pieces.
- Consumer tech, entertainment, gaming, phones, and gadgets.
- Deep technical research with no operator angle.
- Politics or macro news with no AI/business operations relevance.
- Articles with no actionable angle for SMB owners or agency operators.

Return ONLY valid JSON with this shape:
{
  "scores": [
    {
      "id": 1,
      "score": 87,
      "category": "ai_world",
      "reasons": ["SMB workflow relevance", "Clear implementation angle"]
    }
  ]
}

Rules:
- Score every provided id exactly once.
- score must be an integer from 0 to 100.
- category must be one of: "ai_world", "business_world", "teach_this_candidate", "exclude".
- reasons must contain 1 to 3 short phrases.
- Do not include Markdown, comments, or extra text.
"""


class ArticleScoringError(RuntimeError):
    pass


def _raw_response_text(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except TypeError:
        return str(raw)


def _message_content(choice: dict) -> str:
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _record_scoring_incident(stage: str, error: Exception | str, raw: str = "", metadata: dict | None = None) -> Path | None:
    raw_text = _raw_response_text(raw)
    incident = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "model": OPENROUTER_MODEL,
        "error": str(error),
        "metadata": metadata or {},
        "raw_response": raw_text[:50000],
    }

    try:
        incident_dir = Path(NEWSLETTER_INCIDENT_DIR)
        incident_dir.mkdir(parents=True, exist_ok=True)
        path = incident_dir / f"{incident['created_at'].replace(':', '-')}-{incident['id']}.json"
        path.write_text(json.dumps(incident, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.error("[scoring] recorded incident: %s", path)
        return path
    except Exception as exc:
        logger.error("[scoring] failed to record incident: %s", exc)
        return None


def _extract_json(raw: str) -> dict:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("empty model response")
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _format_articles_for_scoring(articles: list[dict]) -> str:
    lines = []
    for index, article in enumerate(articles, start=1):
        title = article.get("title", "")
        url = article.get("url", "")
        source = article.get("_source", "")
        snippet = article.get("snippet", "")
        lines.append(
            "\n".join(
                [
                    f"id={index}",
                    f"title={title}",
                    f"url={url}",
                    f"source={source}",
                    f"snippet={snippet[:500]}",
                ]
            )
        )
    return "\n\n".join(lines)


async def _call_openrouter(client: httpx.AsyncClient, headers: dict, messages: list[dict]) -> dict:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": SCORING_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code == 400 and "response_format" in response.text:
        logger.warning("[scoring] model/provider rejected JSON mode; retrying without response_format")
        payload.pop("response_format", None)
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise ArticleScoringError(f"OpenRouter returned {response.status_code}: {response.text[:500]}")

    return response.json()


def _validate_scores(data: dict, article_count: int) -> dict[int, dict]:
    scores = data.get("scores")
    if not isinstance(scores, list):
        raise ValueError("scores must be a list")

    by_id = {}
    expected_ids = set(range(1, article_count + 1))
    for item in scores:
        if not isinstance(item, dict):
            raise ValueError("score item must be an object")

        article_id = item.get("id")
        score = item.get("score")
        category = item.get("category")
        if isinstance(category, str):
            category = category.strip().lower().replace("-", "_").replace(" ", "_")
            category = CATEGORY_ALIASES.get(category, category)
        reasons = item.get("reasons")

        if article_id not in expected_ids:
            raise ValueError(f"invalid score id: {article_id}")
        if article_id in by_id:
            raise ValueError(f"duplicate score id: {article_id}")
        if not isinstance(score, int) or not 0 <= score <= 100:
            raise ValueError(f"invalid score for id {article_id}: {score}")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category for id {article_id}: {category}")
        if not isinstance(reasons, list) or not 1 <= len(reasons) <= 3:
            raise ValueError(f"invalid reasons for id {article_id}")
        if not all(isinstance(reason, str) and reason.strip() for reason in reasons):
            raise ValueError(f"invalid reason text for id {article_id}")

        by_id[article_id] = {
            "score": score,
            "category": category,
            "reasons": [reason.strip() for reason in reasons],
        }

    missing = expected_ids - set(by_id)
    if missing:
        raise ValueError(f"missing scores for ids: {sorted(missing)[:10]}")

    return by_id


def _apply_scores(articles: list[dict], scores_by_id: dict[int, dict]) -> list[dict]:
    scored = []
    for index, article in enumerate(articles, start=1):
        score_data = scores_by_id[index]
        enriched = dict(article)
        enriched["editorial_score"] = score_data["score"]
        enriched["editorial_category"] = score_data["category"]
        enriched["editorial_reasons"] = score_data["reasons"]
        enriched["_score_input_order"] = index
        scored.append(enriched)

    return sorted(scored, key=lambda article: (-article["editorial_score"], article["_score_input_order"]))


def _heuristic_category(article: dict) -> str:
    text = " ".join(
        str(article.get(field, ""))
        for field in ("title", "snippet", "url", "_source")
    ).lower()

    if any(term in text for term in ("automation", "agent", "workflow", "artificial intelligence", " ai ", "llm", "openai")):
        return "ai_world"
    if any(term in text for term in ("small business", "smb", "startup", "founder", "agency", "sales", "customer", "revenue")):
        return "business_world"
    return "business_world"


def _heuristic_score(article: dict) -> int:
    text = " ".join(
        str(article.get(field, ""))
        for field in ("title", "snippet", "url", "_source")
    ).lower()
    score = 45

    keyword_weights = {
        "small business": 14,
        "smb": 14,
        "automation": 14,
        "workflow": 12,
        "agent": 10,
        "ai": 10,
        "artificial intelligence": 10,
        "founder": 8,
        "agency": 8,
        "sales": 7,
        "support": 7,
        "operations": 7,
        "case study": 7,
        "roi": 6,
        "template": 5,
        "integration": 5,
    }
    for keyword, weight in keyword_weights.items():
        if keyword in text:
            score += weight

    penalties = ("game", "gaming", "phone", "movie", "entertainment", "celebrity")
    for keyword in penalties:
        if keyword in text:
            score -= 12

    return max(0, min(100, score))


def _heuristic_scores(articles: list[dict]) -> list[dict]:
    scored = []
    for index, article in enumerate(articles, start=1):
        enriched = dict(article)
        category = _heuristic_category(article)
        enriched["editorial_score"] = _heuristic_score(article)
        enriched["editorial_category"] = category
        enriched["editorial_reasons"] = ["Heuristic fallback after scoring failure"]
        enriched["_score_input_order"] = index
        scored.append(enriched)

    return sorted(scored, key=lambda article: (-article["editorial_score"], article["_score_input_order"]))


async def score_articles(articles: list[dict], *, client_factory=None) -> list[dict]:
    """Use an LLM to score articles for editorial fit, excluding recency as a factor."""
    if not articles:
        return []
    if not OPENROUTER_API_KEY and client_factory is None:
        raise ArticleScoringError("OPENROUTER_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Daily Newsletter Service",
    }
    user_message = (
        "Score these candidate articles. They are all within the same 30-hour source window, "
        "so do not use publication time as a scoring factor.\n\n"
        f"{_format_articles_for_scoring(articles)}"
    )
    base_messages = [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    async with (client_factory or httpx.AsyncClient)(timeout=HTTPX_TIMEOUT) as client:
        last_error = None
        for attempt in range(1, SCORING_ATTEMPTS + 1):
            messages = list(base_messages)
            if attempt > 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous scoring output failed validation: {last_error}. "
                            "Correct that error and return complete valid JSON only."
                        ),
                    }
                )

            data = await _call_openrouter(client, headers, messages)
            choices = data.get("choices") if isinstance(data, dict) else None
            choice = choices[0] if choices else {}
            raw_content = _message_content(choice)
            finish_reason = choice.get("finish_reason")

            try:
                scores_by_id = _validate_scores(_extract_json(raw_content), len(articles))
                scored = _apply_scores(articles, scores_by_id)
                logger.info("[scoring] scored %s articles with LLM", len(scored))
                return scored
            except Exception as exc:
                last_error = exc
                _record_scoring_incident(
                    "invalid_article_scores",
                    exc,
                    raw_content,
                    {"attempt": attempt, "finish_reason": finish_reason, "article_count": len(articles)},
                )

    _record_scoring_incident(
        "scoring_fallback_used",
        last_error or "Article scoring failed after retries",
        "",
        {"article_count": len(articles), "attempts": SCORING_ATTEMPTS},
    )
    logger.warning("[scoring] falling back to heuristic scores after LLM scoring failed: %s", last_error)
    return _heuristic_scores(articles)
