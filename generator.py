import html
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
PRIMARY_MAX_TOKENS = int(os.getenv("NEWSLETTER_PRIMARY_MAX_TOKENS", "6000"))
REPAIR_MAX_TOKENS = int(os.getenv("NEWSLETTER_REPAIR_MAX_TOKENS", "6000"))
GENERATION_ATTEMPTS = int(os.getenv("NEWSLETTER_GENERATION_ATTEMPTS", "2"))

SYSTEM_PROMPT = """\
You are a sharp, lightly opinionated curator producing a daily AI + business newsletter
for an SMB operator evaluating practical AI workflows.

The reader profile: practitioner, not researcher. 5 minutes max per issue. Core needs:
1. Stay ahead of what SMB clients will ask about next
2. Find tools and workflows worth teaching or adopting immediately
3. Spot content worth sharing on social media

Your newsletter has exactly THREE sections:
1. AI World - AI tools, models, automation workflows, frameworks, and industry shifts
2. Business World - SMB adoption stories, agency/founder news, market moves, and AI policy affecting operators
3. Teach This - ONE story from today's news framed as a lesson for SMB owners

<curation_rules>
INCLUDE:
- AI tools with clear SMB use cases, especially if affordable, easy to learn, or integrated with common business tools
- Real stories of businesses, especially SMBs, implementing AI: case studies, ROI, lessons learned
- News relevant to running or growing an AI automation agency
- AI policy and regulation with practical impact on operators
- Practitioner takes from founders and operators; weight these over journalist/media coverage of the same story

EXCLUDE entirely, even if trending:
- Hype and speculation, including unverified predictions and "AI will replace jobs" think-pieces
- Consumer tech, including phones, gaming, gadgets, and entertainment
- Deep technical research with no practical operator angle
- Politics and macro news unrelated to AI or business operations
- Stories with no actionable angle for an SMB owner or agency operator
</curation_rules>

<output_contract>
Output ONLY valid JSON. No Markdown, no code fence, no HTML, no preamble.

Use this exact shape:
{
  "ai_world": [
    {
      "headline": "Declarative headline statement.",
      "url": "https://source-url-from-input.example",
      "source_label": "Short source name",
      "why_it_matters": "One specific sentence for operators and SMB owners."
    }
  ],
  "business_world": [
    {
      "headline": "Declarative headline statement.",
      "url": "https://source-url-from-input.example",
      "source_label": "Short source name",
      "why_it_matters": "One specific sentence for operators and SMB owners."
    }
  ],
  "teach_this": {
    "headline": "Linked story headline.",
    "url": "https://source-url-from-input.example",
    "source_label": "Short source name",
    "lesson": "Two to three sentences framing the lesson for SMB owners."
  }
}

Content requirements:
- ai_world must contain exactly 5 to 7 items.
- business_world must contain exactly 5 to 7 items.
- For tool stories, why_it_matters must include at least one of: cost, SMB use case, integration fit, or ease of use.
- teach_this must use one story from ai_world or business_world.
- Use only URLs present in the provided source data. Never invent URLs.
- De-duplicate: if the same story appears in multiple sources, cite the best source once.
- Editorial scores: each source may include an editorial_score from 0 to 100 and scoring reasons.
  Use this as a strong ranking signal, but do not blindly follow it if a lower-scored item is clearly more useful or improves section diversity.
- Recency: all supplied sources have already been filtered to the last 30 hours. Do not prefer an item simply because it is newer within that window.
</output_contract>

<writing_rules>
AI-detox: readers must never think "AI wrote this".

BANNED words: delve, realm, tapestry, landscape, leverage, utilize, robust, seamless,
comprehensive, cutting-edge, holistic, synergy, paradigm, empower, innovative, transformative, game-changer.

BANNED phrases: "It's important to note", "In today's X landscape", "Let's dive into",
"Moving forward", "At the end of the day", "When it comes to", "Here's the thing", "This is a game-changer".

Substitutions: leverage->use, utilize->use, implement->build/add, facilitate->help,
in order to->to, due to the fact that->because, a large number of->many, has the ability to->can.

Style:
- Write tight. If a word can be cut without losing meaning, cut it.
- Lead with the fact, not the framing.
- Use active voice. Name the actor.
- Use present tense for headlines.
- One idea per item.
- Use numerals for 10+, money, and percentages. Spell out one through nine.
- No exclamation points. No rhetorical questions.
</writing_rules>
"""

JSON_REPAIR_PROMPT = """\
You repair malformed newsletter JSON.

Return ONLY valid JSON with this exact top-level shape:
{
  "ai_world": [],
  "business_world": [],
  "teach_this": {}
}

Rules:
- Preserve the original meaning where possible.
- Fix broken escaping, unterminated strings, stray Markdown, and trailing text.
- Do not add commentary, Markdown, or code fences.
- If a value cannot be recovered, use a concise replacement based only on the provided source data.
"""


class NewsletterGenerationError(RuntimeError):
    pass


def _escape(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _format_articles(articles: list[dict]) -> str:
    """Format article list into a compact block for the prompt."""
    lines = []
    for article in articles:
        title = _escape(article.get("title", ""))
        url = _escape(article.get("url", ""))
        published_at = _escape(article.get("published_at", ""))
        source = _escape(article.get("_source", ""))
        editorial_score = article.get("editorial_score")
        editorial_category = _escape(article.get("editorial_category", ""))
        editorial_reasons = article.get("editorial_reasons", [])
        score_label = editorial_score if isinstance(editorial_score, int) else "unscored"
        line = (
            f"- {title} | {url} | published_at={published_at} | source={source} "
            f"| editorial_score={score_label} | editorial_category={editorial_category}"
        )
        if editorial_reasons:
            reasons = "; ".join(_escape(reason) for reason in editorial_reasons[:3])
            line += f"\n  Score reasons: {reasons}"
        snippet = article.get("snippet", "")
        if snippet and snippet != article.get("title", ""):
            line += f"\n  {_escape(snippet)}"
        lines.append(line)
    return "\n".join(lines)


def _build_user_message(content: dict) -> str:
    articles = content.get("articles", [])
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y").replace(" 0", " ")
    source_window_hours = content.get("source_window_hours", 30)

    sections = {
        "tavily_results": [a for a in articles if a.get("_source") == "tavily"],
        "reddit_posts": [a for a in articles if a.get("_source") == "reddit"],
        "hackernews_stories": [a for a in articles if a.get("_source") == "hackernews"],
        "medium_articles": [a for a in articles if a.get("_source") == "medium"],
    }
    all_articles = articles if not any(sections.values()) else []

    parts = [
        f"Today is {today}. Generate the newsletter JSON from the sources below.",
        f"All sources have been filtered to the last {source_window_hours} hours.",
        "",
        "<sources>",
    ]

    for section_name, section_articles in sections.items():
        if section_articles:
            parts.append(f"  <{section_name}>")
            parts.append(_format_articles(section_articles))
            parts.append(f"  </{section_name}>")

    if all_articles:
        parts.append("  <all_sources>")
        parts.append(_format_articles(all_articles))
        parts.append("  </all_sources>")

    parts.append("</sources>")
    return "\n".join(parts)


def _extract_json(raw: str) -> dict:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("empty model response")
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


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


def _record_generation_incident(stage: str, error: Exception | str, raw: str = "", metadata: dict | None = None) -> Path | None:
    """Persist enough context to debug model/output failures after a Modal run ends."""
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
        logger.error("[generator] recorded incident: %s", path)
        return path
    except Exception as exc:
        logger.error("[generator] failed to record incident: %s", exc)
        return None


def _canonical_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"source", "ref", "ref_src"}
    ]
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _source_url_lookup(source_urls: set[str]) -> dict[str, str]:
    return {_canonical_url(url): url for url in source_urls}


def _validate_story(item: dict, section: str, source_urls: set[str], source_lookup: dict[str, str]) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{section} item must be an object")

    required = ("headline", "url", "source_label", "why_it_matters")
    for field in required:
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"{section} item missing {field}")

    if item["url"] in source_urls:
        return

    canonical = _canonical_url(item["url"])
    if canonical in source_lookup:
        item["url"] = source_lookup[canonical]
        return

    if item["url"].startswith("https://medium.com/") and item["url"].split("?")[0] in source_lookup:
        item["url"] = source_lookup[item["url"].split("?")[0]]
        return

    else:
        raise ValueError(f"{section} item uses URL outside source data: {item['url']}")


def _validate_issue(issue: dict, source_urls: set[str]) -> dict:
    if not isinstance(issue, dict):
        raise ValueError("newsletter issue must be an object")

    ai_items = issue.get("ai_world")
    business_items = issue.get("business_world")
    teach = issue.get("teach_this")

    if not isinstance(ai_items, list) or not 5 <= len(ai_items) <= 7:
        raise ValueError("ai_world must contain 5 to 7 items")
    if not isinstance(business_items, list) or not 5 <= len(business_items) <= 7:
        raise ValueError("business_world must contain 5 to 7 items")

    source_lookup = _source_url_lookup(source_urls)
    for item in ai_items:
        _validate_story(item, "ai_world", source_urls, source_lookup)
    for item in business_items:
        _validate_story(item, "business_world", source_urls, source_lookup)

    if not isinstance(teach, dict):
        raise ValueError("teach_this must be an object")
    for field in ("headline", "url", "source_label", "lesson"):
        if not isinstance(teach.get(field), str) or not teach[field].strip():
            raise ValueError(f"teach_this missing {field}")

    if teach["url"] not in source_urls:
        canonical_teach_url = _canonical_url(teach["url"])
        if canonical_teach_url in source_lookup:
            teach["url"] = source_lookup[canonical_teach_url]

    if teach["url"] not in {item["url"] for item in ai_items + business_items}:
        raise ValueError("teach_this must use a story from ai_world or business_world")

    return issue


def _source_label(article: dict) -> str:
    source = (article.get("_source") or "").strip().lower()
    labels = {
        "hackernews": "Hacker News",
        "medium": "Medium",
        "reddit": "Reddit",
        "tavily": "News",
    }
    if source in labels:
        return labels[source]

    netloc = urlsplit(article.get("url", "")).netloc.replace("www.", "")
    return netloc or "Source"


def _fallback_why(section: str) -> str:
    if section == "ai_world":
        return "This gives SMB owners a practical AI signal to test against real workflows."
    return "This shows operators where market, customer, or workflow expectations are moving."


def _story_from_article(article: dict, section: str) -> dict:
    return {
        "headline": str(article.get("title") or "Useful operator signal").strip(),
        "url": str(article.get("url") or "").strip(),
        "source_label": _source_label(article),
        "why_it_matters": _fallback_why(section),
    }


def _article_matches_section(article: dict, section: str) -> bool:
    category = article.get("editorial_category")
    if category == section:
        return True
    if section == "ai_world" and category == "teach_this_candidate":
        return True
    return False


def _candidate_articles(articles: list[dict], section: str) -> list[dict]:
    preferred = [article for article in articles if _article_matches_section(article, section)]
    usable = [article for article in articles if article.get("url") and article.get("editorial_category") != "exclude"]
    fallback = [article for article in articles if article.get("url")]

    ordered = []
    seen = set()
    for article in preferred + usable + fallback:
        url = article.get("url")
        if url and url not in seen:
            ordered.append(article)
            seen.add(url)
    return ordered


def _validated_story_or_none(item: dict, section: str, source_urls: set[str], source_lookup: dict[str, str]) -> dict | None:
    if not isinstance(item, dict):
        return None

    candidate = {
        "headline": str(item.get("headline") or "").strip(),
        "url": str(item.get("url") or "").strip(),
        "source_label": str(item.get("source_label") or "").strip(),
        "why_it_matters": str(item.get("why_it_matters") or "").strip(),
    }
    try:
        _validate_story(candidate, section, source_urls, source_lookup)
        return candidate
    except ValueError:
        return None


def _recover_section(
    items: list,
    section: str,
    articles: list[dict],
    source_urls: set[str],
    source_lookup: dict[str, str],
    used_urls: set[str],
) -> list[dict]:
    original_len = len(items) if isinstance(items, list) else 0
    target_len = min(max(original_len, 5), 7)
    recovered = []

    if isinstance(items, list):
        for item in items:
            story = _validated_story_or_none(item, section, source_urls, source_lookup)
            if story and story["url"] not in used_urls:
                recovered.append(story)
                used_urls.add(story["url"])
            if len(recovered) == target_len:
                return recovered

    for article in _candidate_articles(articles, section):
        if article["url"] in used_urls:
            continue
        story = _story_from_article(article, section)
        if story["url"] in source_urls:
            recovered.append(story)
            used_urls.add(story["url"])
        if len(recovered) == target_len:
            return recovered

    if len(recovered) < 5:
        raise ValueError(f"not enough source-backed stories to recover {section}")

    return recovered


def _recover_issue_from_sources(issue: dict, articles: list[dict], source_urls: set[str]) -> dict:
    """Replace invalid model-selected stories with deterministic source-backed items."""
    source_lookup = _source_url_lookup(source_urls)
    used_urls: set[str] = set()

    recovered = {
        "ai_world": _recover_section(
            issue.get("ai_world") if isinstance(issue, dict) else [],
            "ai_world",
            articles,
            source_urls,
            source_lookup,
            used_urls,
        ),
        "business_world": _recover_section(
            issue.get("business_world") if isinstance(issue, dict) else [],
            "business_world",
            articles,
            source_urls,
            source_lookup,
            used_urls,
        ),
    }

    teach = issue.get("teach_this") if isinstance(issue, dict) else {}
    section_items = recovered["ai_world"] + recovered["business_world"]
    teach_url = teach.get("url") if isinstance(teach, dict) else None
    teach_item = next((item for item in section_items if item["url"] == teach_url), None)
    if teach_item is None and isinstance(teach, dict):
        canonical_teach_url = _canonical_url(str(teach_url or ""))
        teach_item = next((item for item in section_items if _canonical_url(item["url"]) == canonical_teach_url), None)

    if teach_item is None:
        teach_item = section_items[0]

    lesson = teach.get("lesson") if isinstance(teach, dict) else ""
    if not isinstance(lesson, str) or not lesson.strip():
        lesson = "Start with one concrete workflow, then measure the before and after. The useful lesson is the operating pattern, not the headline itself."

    recovered["teach_this"] = {
        "headline": teach_item["headline"],
        "url": teach_item["url"],
        "source_label": teach_item["source_label"],
        "lesson": lesson.strip(),
    }
    return _validate_issue(recovered, source_urls)


async def _call_openrouter(client: httpx.AsyncClient, headers: dict, messages: list[dict], max_tokens: int) -> dict:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code == 400 and "response_format" in response.text:
        logger.warning("[generator] model/provider rejected JSON mode; retrying without response_format")
        payload.pop("response_format", None)
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text[:500]}")

    return response.json()


async def _repair_issue_json(
    client: httpx.AsyncClient,
    headers: dict,
    raw: str,
    parse_error: Exception,
    user_message: str,
    source_urls: set[str],
    articles: list[dict],
) -> dict:
    logger.warning("[generator] repairing malformed model JSON: %s", parse_error)
    repair_messages = [
        {"role": "system", "content": JSON_REPAIR_PROMPT},
        {
            "role": "user",
            "content": (
                f"The previous newsletter JSON failed to parse with this error:\n{parse_error}\n\n"
                f"Malformed response:\n{raw}\n\n"
                f"Source data, for URL validation and missing values:\n{user_message}"
            ),
        },
    ]
    data = await _call_openrouter(client, headers, repair_messages, REPAIR_MAX_TOKENS)
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if choices else {}
    repaired_raw = _message_content(choice)
    try:
        repaired_issue = _extract_json(repaired_raw)
        try:
            return _validate_issue(repaired_issue, source_urls)
        except Exception:
            return _recover_issue_from_sources(repaired_issue, articles, source_urls)
    except Exception as exc:
        _record_generation_incident(
            "json_repair_failed",
            exc,
            repaired_raw,
            {"original_error": str(parse_error), "finish_reason": choice.get("finish_reason")},
        )
        raise


def _item_html(item: dict, accent: str) -> str:
    headline = _escape(item.get("headline"))
    url = _escape(item.get("url"))
    source_label = _escape(item.get("source_label") or "Source")
    why = _escape(item.get("why_it_matters"))
    return f"""
      <tr>
        <td style="padding:18px 0;border-top:1px solid #e6ebf2;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
              <td width="10" valign="top" style="padding-top:7px;">
                <div style="width:6px;height:6px;border-radius:6px;background:{accent};font-size:0;line-height:0;">&nbsp;</div>
              </td>
              <td style="padding-left:10px;">
                <a href="{url}" style="color:#172033;text-decoration:none;font-size:17px;line-height:1.35;font-weight:700;">{headline}</a>
                <div style="padding-top:6px;font-size:12px;line-height:1.3;color:#667085;text-transform:uppercase;letter-spacing:0.08em;">{source_label}</div>
                <p style="margin:10px 0 0;color:#344054;font-size:15px;line-height:1.55;"><strong style="color:#172033;">Why it matters:</strong> {why}</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>"""


def _section_html(title: str, items: list[dict], accent: str) -> str:
    item_rows = "\n".join(_item_html(item, accent) for item in items)
    return f"""
    <tr>
      <td style="padding:28px 30px 0;">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
          <tr>
            <td style="border-bottom:2px solid {accent};padding-bottom:8px;">
              <h2 style="margin:0;color:#172033;font-size:18px;line-height:1.25;font-weight:800;">{_escape(title)}</h2>
            </td>
          </tr>
          {item_rows}
        </table>
      </td>
    </tr>"""


def _render_email(issue: dict, subject: str, source_window_hours: int) -> str:
    date_label = subject.replace("Daily Brief - ", "")
    ai_items = issue.get("ai_world", [])
    business_items = issue.get("business_world", [])
    teach = issue.get("teach_this", {})
    teach_headline = _escape(teach.get("headline"))
    teach_url = _escape(teach.get("url"))
    teach_source = _escape(teach.get("source_label") or "Source")
    teach_lesson = _escape(teach.get("lesson"))

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape(subject)}</title>
  </head>
  <body style="margin:0;padding:0;background:#f5f7fb;color:#172033;font-family:Arial,Helvetica,sans-serif;">
    <div style="display:none;max-height:0;overflow:hidden;color:transparent;opacity:0;">A practical AI and business brief for operators, filtered to the last {source_window_hours} hours.</div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f5f7fb;">
      <tr>
        <td align="center" style="padding:28px 14px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="max-width:640px;background:#ffffff;border:1px solid #d8dee9;border-radius:8px;overflow:hidden;">
            <tr>
              <td style="padding:28px 30px 24px;border-top:5px solid #2454ff;background:#ffffff;">
                <div style="font-size:12px;line-height:1.3;color:#667085;text-transform:uppercase;letter-spacing:0.12em;">{_escape(date_label)}</div>
                <h1 style="margin:8px 0 0;color:#172033;font-size:30px;line-height:1.15;font-weight:800;">Daily Brief</h1>
                <p style="margin:10px 0 0;color:#475467;font-size:15px;line-height:1.55;">AI and business signals for SMB operators, agency builders, and practical automation work.</p>
              </td>
            </tr>
            {_section_html("AI World", ai_items, "#12a594")}
            {_section_html("Business World", business_items, "#d99000")}
            <tr>
              <td style="padding:30px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f7f4ff;border:1px solid #ddd6fe;border-radius:8px;">
                  <tr>
                    <td style="padding:22px 22px 24px;">
                      <div style="font-size:12px;line-height:1.3;color:#7c3aed;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;">Teach This</div>
                      <h2 style="margin:8px 0 0;font-size:19px;line-height:1.35;font-weight:800;"><a href="{teach_url}" style="color:#172033;text-decoration:none;">{teach_headline}</a></h2>
                      <div style="padding-top:6px;font-size:12px;line-height:1.3;color:#667085;text-transform:uppercase;letter-spacing:0.08em;">{teach_source}</div>
                      <p style="margin:12px 0 0;color:#344054;font-size:15px;line-height:1.6;">{teach_lesson}</p>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr>
              <td style="padding:0 30px 28px;">
                <p style="margin:0;padding-top:18px;border-top:1px solid #e6ebf2;color:#667085;font-size:12px;line-height:1.5;">Curated from sources published in the last {source_window_hours} hours. Generated for Vatsal's AI and SMB operator workflow.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""


def _html_to_text(markup: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", markup)
    text = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|td|section|article)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line and line != "&nbsp;")


async def generate_newsletter(content: dict, *, client_factory=None) -> dict:
    """Call OpenRouter for content, then render a designed HTML newsletter."""
    if not OPENROUTER_API_KEY and client_factory is None:
        raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")

    today = datetime.now(timezone.utc).strftime("%B %d, %Y").replace(" 0", " ")
    subject = f"Daily Brief - {today}"
    user_message = _build_user_message(content)

    logger.info(f"[generator] calling OpenRouter model {OPENROUTER_MODEL}...")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Daily Newsletter Service",
    }
    source_urls = {article["url"] for article in content.get("articles", []) if article.get("url")}
    articles = content.get("articles", [])
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    async with (client_factory or httpx.AsyncClient)(timeout=HTTPX_TIMEOUT) as client:
        last_error = None
        for attempt in range(1, GENERATION_ATTEMPTS + 1):
            messages = list(base_messages)
            if attempt > 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous attempt failed validation. Generate the complete newsletter again as "
                            "strict JSON only. Make every string properly escaped and use only source URLs."
                        ),
                    }
                )

            data = await _call_openrouter(client, headers, messages, PRIMARY_MAX_TOKENS)
            choices = data.get("choices") if isinstance(data, dict) else None
            choice = choices[0] if choices else {}
            raw_content = _message_content(choice)
            finish_reason = choice.get("finish_reason")

            if finish_reason == "length":
                last_error = ValueError("OpenRouter response ended because max_tokens was reached")
                _record_generation_incident(
                    "generation_truncated",
                    last_error,
                    raw_content,
                    {"attempt": attempt, "finish_reason": finish_reason},
                )
                continue

            try:
                parsed_issue = _extract_json(raw_content)
                try:
                    issue = _validate_issue(parsed_issue, source_urls)
                except Exception as validation_exc:
                    issue = _recover_issue_from_sources(parsed_issue, articles, source_urls)
                    _record_generation_incident(
                        "invalid_issue_recovered",
                        validation_exc,
                        raw_content,
                        {"attempt": attempt, "finish_reason": finish_reason},
                    )
                break
            except json.JSONDecodeError as exc:
                _record_generation_incident(
                    "invalid_json",
                    exc,
                    raw_content,
                    {"attempt": attempt, "finish_reason": finish_reason},
                )
                try:
                    issue = await _repair_issue_json(client, headers, raw_content, exc, user_message, source_urls, articles)
                    break
                except Exception as repair_exc:
                    last_error = repair_exc
                    continue
            except Exception as exc:
                last_error = exc
                _record_generation_incident(
                    "invalid_issue",
                    exc,
                    raw_content,
                    {"attempt": attempt, "finish_reason": finish_reason},
                )
                continue
        else:
            raise NewsletterGenerationError(f"Newsletter generation failed after retries: {last_error}")

    html_body = _render_email(issue, subject, content.get("source_window_hours", 30))
    text_body = _html_to_text(html_body)

    usage = data.get("usage", {})
    logger.info(
        "[generator] done. tokens - prompt: %s, completion: %s",
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )

    return {
        "subject": subject,
        "html": html_body,
        "text": text_body,
    }
