import asyncio
import unittest
from unittest.mock import patch

import httpx
from offline_demo import FixtureProvider, run_demo


class OfflineBoundaryTests(unittest.TestCase):
    def test_demo_runs_without_network_or_credentials(self):
        with patch("socket.create_connection", side_effect=AssertionError("Network is forbidden")), \
             patch("generator.OPENROUTER_API_KEY", None), patch("scoring.OPENROUTER_API_KEY", None):
            result = asyncio.run(run_demo())
        self.assertEqual(result["articles_scored"], 10)
        self.assertEqual(result["model_calls_mocked"], 2)
        self.assertEqual(result["emails_sent"], 0)
        self.assertIn("Synthetic fixture demo", result["newsletter"]["html"])

    def test_malformed_generation_is_repaired_from_fixture(self):
        with patch("generator.NEWSLETTER_INCIDENT_DIR", "output/test-incidents"):
            result = asyncio.run(run_demo(malformed=True))
        self.assertEqual(result["model_calls_mocked"], 3)
        self.assertIn("Example", result["newsletter"]["text"])

    def test_fixture_adapter_rejects_unknown_endpoint(self):
        provider = FixtureProvider([{}])
        with self.assertRaises(AssertionError):
            provider.handle(httpx.Request("POST", "https://example.com/unexpected"))

    def test_live_client_requires_a_key(self):
        import scoring
        with patch("scoring.OPENROUTER_API_KEY", None):
            with self.assertRaises(scoring.ArticleScoringError):
                asyncio.run(scoring.score_articles([{"title": "Example", "url": "https://example.com"}]))
