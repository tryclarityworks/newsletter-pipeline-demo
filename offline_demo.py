"""Exercise the real scoring/generation validators through an in-memory HTTP adapter."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

import generator
import scoring

ROOT = Path(__file__).resolve().parent


class FixtureProvider:
    """Every request is handled locally; unknown endpoints and excess calls fail."""

    def __init__(self, responses: list[dict | str]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) != "https://openrouter.ai/api/v1/chat/completions":
            raise AssertionError("The offline adapter received an unexpected endpoint")
        if not self.responses:
            raise AssertionError("The offline adapter received an unexpected extra call")
        payload = json.loads(request.content)
        self.requests.append({"message_count": len(payload["messages"]), "mocked": True})
        content = self.responses.pop(0)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content if isinstance(content, str) else json.dumps(content)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        })

    def client(self, **kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), **kwargs)


async def run_demo(*, malformed: bool = False) -> dict:
    articles = json.loads((ROOT / "examples/articles.json").read_text())
    issue = json.loads((ROOT / "examples/model-issue.json").read_text())
    score_response = {"scores": [{
        "id": i + 1, "score": 90 - i,
        "category": "ai_world" if i < 5 else "business_world",
        "reasons": ["Synthetic fixture relevance for an SMB operator"],
    } for i in range(len(articles))]}
    responses = [score_response, '{"ai_world": [unterminated' if malformed else issue]
    if malformed:
        responses.append(issue)
    provider = FixtureProvider(responses)
    scored = await scoring.score_articles(articles, client_factory=provider.client)
    newsletter = await generator.generate_newsletter(
        {"articles": scored, "source_window_hours": 30}, client_factory=provider.client,
    )
    banner = '<p style="padding:16px;background:#fff3cd;color:#332701">Synthetic fixture demo. No live news, model call, or email delivery.</p>'
    # Insert inside the opening body tag, including its existing attributes.
    import re
    newsletter["html"] = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + banner, newsletter["html"], count=1)
    return {"mode": "offline fixtures", "model_calls_mocked": len(provider.requests),
            "articles_scored": len(scored), "malformed_response_exercised": malformed,
            "emails_sent": 0, "newsletter": newsletter}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--malformed", action="store_true", help="Exercise malformed JSON and the repair path")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    result = asyncio.run(run_demo(malformed=args.malformed))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "newsletter.html").write_text(result["newsletter"]["html"])
    (args.output / "newsletter.txt").write_text(result["newsletter"]["text"])
    (args.output / "run.json").write_text(json.dumps({k: v for k, v in result.items() if k != "newsletter"}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "newsletter"}, indent=2))
    print(f"Preview: {args.output / 'newsletter.html'}")


if __name__ == "__main__":
    main()
