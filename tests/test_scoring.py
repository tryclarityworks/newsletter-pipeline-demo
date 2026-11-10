import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import scoring


class FakeResponse:
    def __init__(self, content):
        self.status_code = 200
        self.text = "" if content is None else str(content)
        self._content = content

    def json(self):
        return {
            "choices": [
                {
                    "message": {"content": self._content},
                    "finish_reason": "stop",
                }
            ]
        }


class FakeClient:
    def __init__(self, response: str | list[str]):
        self.responses = response if isinstance(response, list) else [response]
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        self.requests.append(kwargs["json"])
        response_index = min(len(self.requests) - 1, len(self.responses) - 1)
        return FakeResponse(self.responses[response_index])


class ScoringTests(unittest.TestCase):
    def test_score_articles_uses_llm_scores_without_recency_tiebreak(self):
        original_client = scoring.httpx.AsyncClient
        original_key = scoring.OPENROUTER_API_KEY

        response = json.dumps(
            {
                "scores": [
                    {
                        "id": 1,
                        "score": 40,
                        "category": "exclude",
                        "reasons": ["Low operator relevance"],
                    },
                    {
                        "id": 2,
                        "score": 95,
                        "category": "ai_world",
                        "reasons": ["Strong SMB workflow use case"],
                    },
                ]
            }
        )
        fake_client = FakeClient(response)
        scoring.httpx.AsyncClient = lambda *args, **kwargs: fake_client
        scoring.OPENROUTER_API_KEY = "test-key"

        articles = [
            {
                "title": "Newer generic AI opinion",
                "url": "https://example.com/newer",
                "snippet": "A broad prediction piece.",
                "published_at": "2026-05-28T20:00:00+00:00",
                "_source": "tavily",
            },
            {
                "title": "Older SMB automation case study",
                "url": "https://example.com/older",
                "snippet": "A small business implementation story.",
                "published_at": "2026-05-27T20:00:00+00:00",
                "_source": "tavily",
            },
        ]

        try:
            scored = asyncio.run(scoring.score_articles(articles))
        finally:
            scoring.httpx.AsyncClient = original_client
            scoring.OPENROUTER_API_KEY = original_key

        self.assertEqual(scored[0]["url"], "https://example.com/older")
        self.assertEqual(scored[0]["editorial_score"], 95)
        self.assertEqual(scored[1]["editorial_score"], 40)
        prompt = fake_client.requests[0]["messages"][0]["content"]
        self.assertIn("Do NOT score based on recency", prompt)

    def test_validate_scores_requires_every_article_id(self):
        with self.assertRaises(ValueError):
            scoring._validate_scores({"scores": [{"id": 1, "score": 90, "category": "ai_world", "reasons": ["Useful"]}]}, 2)

    def test_validate_scores_normalizes_automation_category(self):
        scores = scoring._validate_scores(
            {"scores": [{"id": 1, "score": 90, "category": "automation", "reasons": ["Useful"]}]},
            1,
        )

        self.assertEqual(scores[1]["category"], "ai_world")

    def test_score_articles_includes_validation_error_in_retry(self):
        original_attempts = scoring.SCORING_ATTEMPTS
        original_client = scoring.httpx.AsyncClient
        original_key = scoring.OPENROUTER_API_KEY
        invalid = json.dumps(
            {"scores": [{"id": 1, "score": 90, "category": "other", "reasons": ["Useful"]}]}
        )
        valid = json.dumps(
            {"scores": [{"id": 1, "score": 90, "category": "ai_world", "reasons": ["Useful"]}]}
        )
        fake_client = FakeClient([invalid, valid])
        scoring.SCORING_ATTEMPTS = 2
        scoring.httpx.AsyncClient = lambda *args, **kwargs: fake_client
        scoring.OPENROUTER_API_KEY = "test-key"

        try:
            asyncio.run(scoring.score_articles([{"title": "AI story", "url": "https://example.com"}]))
        finally:
            scoring.SCORING_ATTEMPTS = original_attempts
            scoring.httpx.AsyncClient = original_client
            scoring.OPENROUTER_API_KEY = original_key

        retry_message = fake_client.requests[1]["messages"][-1]["content"]
        self.assertIn("invalid category for id 1: other", retry_message)

    def test_score_articles_falls_back_when_model_content_is_none(self):
        original_attempts = scoring.SCORING_ATTEMPTS
        original_client = scoring.httpx.AsyncClient
        original_key = scoring.OPENROUTER_API_KEY
        original_incident_dir = scoring.NEWSLETTER_INCIDENT_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = FakeClient([None, None])
            scoring.SCORING_ATTEMPTS = 2
            scoring.httpx.AsyncClient = lambda *args, **kwargs: fake_client
            scoring.OPENROUTER_API_KEY = "test-key"
            scoring.NEWSLETTER_INCIDENT_DIR = tmpdir

            articles = [
                {
                    "title": "SMB automation workflow example",
                    "url": "https://example.com/automation",
                    "snippet": "A small business uses AI agents to reduce support work.",
                    "_source": "tavily",
                }
            ]

            try:
                scored = asyncio.run(scoring.score_articles(articles))
            finally:
                scoring.SCORING_ATTEMPTS = original_attempts
                scoring.httpx.AsyncClient = original_client
                scoring.OPENROUTER_API_KEY = original_key
                scoring.NEWSLETTER_INCIDENT_DIR = original_incident_dir

            self.assertEqual(scored[0]["url"], "https://example.com/automation")
            self.assertEqual(scored[0]["editorial_category"], "ai_world")
            self.assertIn("Heuristic fallback", scored[0]["editorial_reasons"][0])

            incident_files = list(Path(tmpdir).glob("*.json"))
            self.assertGreaterEqual(len(incident_files), 3)
            incident_text = "\n".join(path.read_text() for path in incident_files)
            self.assertIn("empty model response", incident_text)
            self.assertIn("scoring_fallback_used", incident_text)


if __name__ == "__main__":
    unittest.main()
