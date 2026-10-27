import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import get_prospect, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "REP-001"}})
        self.rep = {"name": "Rep", "email": "rep@example.com"}

    def send(self, prospect):
        return send_prospect_email.func(
            prospect, "Hello", "Let's talk.", self.runtime, from_rep=self.rep,
        )

    def test_disqualified_source_blocks_even_if_contact_claims_eligible(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-50001"})["prospect"]
        prospect["disqualified"] = False
        with patch("gtm_agent.gtm_agent.uuid.uuid4", side_effect=AssertionError("Cannot send")):
            result = self.send(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": prospect["email"],
        })

    def test_eligible_source_preserves_success_result(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
        result = self.send(prospect)
        self.assertRegex(result.pop("message_id"), r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result, {
            "status": "sent", "to": prospect["email"], "to_name": prospect["name"],
            "from": self.rep["email"], "from_name": "Rep",
            "subject": "Hello", "body": "Let's talk.",
        })

    def test_missing_source_or_id_blocks_without_claiming_disqualification(self):
        for prospect in ({"prospect_id": "unknown", "email": "prospect@example.com"},
                         {"email": "prospect@example.com"}):
            with self.subTest(prospect=prospect):
                with patch("gtm_agent.gtm_agent.uuid.uuid4", side_effect=AssertionError("Cannot send")):
                    result = self.send(prospect)
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect record could not be found; email not sent.",
                    "to": prospect["email"],
                })

    def test_missing_email_preserves_failure_for_any_source(self):
        for prospect_id in ("LEAD-50001", "unknown", None):
            with self.subTest(prospect_id=prospect_id):
                with patch("gtm_agent.gtm_agent.uuid.uuid4", side_effect=AssertionError("Cannot send")):
                    result = self.send({"prospect_id": prospect_id})
                self.assertEqual(result, {
                    "status": "failed", "error": "Prospect record has no email address.",
                })


if __name__ == "__main__":
    unittest.main()
