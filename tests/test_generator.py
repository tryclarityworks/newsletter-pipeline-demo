import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import generator


def _valid_issue() -> dict:
    ai_items = [
        {
            "headline": f"AI operator story {i}",
            "url": f"https://example.com/ai-{i}",
            "source_label": "Example",
            "why_it_matters": "This gives SMB owners a practical workflow to test this week.",
        }
        for i in range(5)
    ]
    business_items = [
        {
            "headline": f"Business operator story {i}",
            "url": f"https://example.com/business-{i}",
            "source_label": "Example",
            "why_it_matters": "This shows owners where AI can cut manual work without a large budget.",
        }
        for i in range(5)
    ]
    return {
        "ai_world": ai_items,
        "business_world": business_items,
        "teach_this": {
            "headline": ai_items[0]["headline"],
            "url": ai_items[0]["url"],
            "source_label": "Example",
            "lesson": "Start with one repeated workflow. Measure the before and after before expanding.",
        },
    }


def _content() -> dict:
    issue = _valid_issue()
    urls = [item["url"] for item in issue["ai_world"] + issue["business_world"]]
    return {
        "source_window_hours": 30,
        "articles": [
            {
                "title": f"Source {i}",
                "url": url,
                "snippet": "A useful source item.",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "_source": "tavily",
            }
            for i, url in enumerate(urls)
        ],
    }


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
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        if not self.responses:
            raise AssertionError("No fake responses left")
        return FakeResponse(self.responses.pop(0))


class GeneratorTests(unittest.TestCase):
    def test_generate_newsletter_repairs_malformed_json(self):
        original_client = generator.httpx.AsyncClient
        original_key = generator.OPENROUTER_API_KEY
        original_incident_dir = generator.NEWSLETTER_INCIDENT_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            malformed = '{"ai_world": [{"headline": "unterminated]'
            repaired = json.dumps(_valid_issue())
            fake_client = FakeClient([malformed, repaired])

            generator.httpx.AsyncClient = lambda *args, **kwargs: fake_client
            generator.OPENROUTER_API_KEY = "test-key"
            generator.NEWSLETTER_INCIDENT_DIR = tmpdir

            try:
                newsletter = asyncio.run(generator.generate_newsletter(_content()))
            finally:
                generator.httpx.AsyncClient = original_client
                generator.OPENROUTER_API_KEY = original_key
                generator.NEWSLETTER_INCIDENT_DIR = original_incident_dir

            self.assertIn("Daily Brief", newsletter["subject"])
            self.assertIn("AI operator story 0", newsletter["html"])
            self.assertTrue(list(Path(tmpdir).glob("*.json")))

    def test_generate_newsletter_retries_when_model_content_is_none(self):
        original_client = generator.httpx.AsyncClient
        original_key = generator.OPENROUTER_API_KEY
        original_incident_dir = generator.NEWSLETTER_INCIDENT_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = FakeClient([None, json.dumps(_valid_issue())])

            generator.httpx.AsyncClient = lambda *args, **kwargs: fake_client
            generator.OPENROUTER_API_KEY = "test-key"
            generator.NEWSLETTER_INCIDENT_DIR = tmpdir

            try:
                newsletter = asyncio.run(generator.generate_newsletter(_content()))
            finally:
                generator.httpx.AsyncClient = original_client
                generator.OPENROUTER_API_KEY = original_key
                generator.NEWSLETTER_INCIDENT_DIR = original_incident_dir

            self.assertIn("Daily Brief", newsletter["subject"])
            self.assertIn("AI operator story 0", newsletter["html"])
            incident_text = "\n".join(path.read_text() for path in Path(tmpdir).glob("*.json"))
            self.assertIn("empty model response", incident_text)

    def test_generate_newsletter_recovers_model_url_outside_source_data(self):
        original_client = generator.httpx.AsyncClient
        original_key = generator.OPENROUTER_API_KEY
        original_incident_dir = generator.NEWSLETTER_INCIDENT_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            issue = _valid_issue()
            issue["business_world"][0]["url"] = "https://news.ycombinator.com/item?id=38329446"
            fake_client = FakeClient([json.dumps(issue)])

            generator.httpx.AsyncClient = lambda *args, **kwargs: fake_client
            generator.OPENROUTER_API_KEY = "test-key"
            generator.NEWSLETTER_INCIDENT_DIR = tmpdir

            try:
                newsletter = asyncio.run(generator.generate_newsletter(_content()))
            finally:
                generator.httpx.AsyncClient = original_client
                generator.OPENROUTER_API_KEY = original_key
                generator.NEWSLETTER_INCIDENT_DIR = original_incident_dir

            self.assertIn("Daily Brief", newsletter["subject"])
            self.assertNotIn("38329446", newsletter["html"])
            self.assertIn("https://example.com/business-0", newsletter["html"])
            self.assertTrue(list(Path(tmpdir).glob("*.json")))

    def test_validate_issue_rejects_urls_outside_source_data(self):
        issue = _valid_issue()
        issue["ai_world"][0]["url"] = "https://not-from-input.example"

        with self.assertRaises(ValueError):
            generator._validate_issue(issue, {"https://example.com/ai-0"})

    def test_validate_issue_accepts_canonicalized_source_urls(self):
        issue = _valid_issue()
        source_url = "https://medium.com/@writer/story-title-123?source=rss&utm_medium=email"
        issue["ai_world"][0]["url"] = "https://medium.com/@writer/story-title-123"
        issue["teach_this"]["url"] = "https://medium.com/@writer/story-title-123"

        source_urls = {item["url"] for item in issue["ai_world"] + issue["business_world"]}
        source_urls.remove("https://medium.com/@writer/story-title-123")
        source_urls.add(source_url)

        validated = generator._validate_issue(issue, source_urls)

        self.assertEqual(validated["ai_world"][0]["url"], source_url)
        self.assertEqual(validated["teach_this"]["url"], source_url)

    def test_format_articles_includes_editorial_score_metadata(self):
        formatted = generator._format_articles(
            [
                {
                    "title": "Useful SMB AI story",
                    "url": "https://example.com/story",
                    "published_at": "2026-05-28T12:00:00+00:00",
                    "_source": "tavily",
                    "snippet": "A practical implementation example.",
                    "editorial_score": 91,
                    "editorial_category": "ai_world",
                    "editorial_reasons": ["SMB workflow relevance", "Clear implementation angle"],
                }
            ]
        )

        self.assertIn("editorial_score=91", formatted)
        self.assertIn("editorial_category=ai_world", formatted)
        self.assertIn("Score reasons: SMB workflow relevance; Clear implementation angle", formatted)


if __name__ == "__main__":
    unittest.main()
