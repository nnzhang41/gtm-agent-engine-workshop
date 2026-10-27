import json
import os
import unittest

from dotenv import load_dotenv
from langsmith import tracing_context

load_dotenv()


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires OpenAI credentials")
class EmailBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gtm_agent.gtm_agent import run_agent

        cls.run_agent = staticmethod(run_agent)

    def send_request(self, request):
        with tracing_context(enabled=False):
            return self.run_agent(request, user_id="rep_sbrown")

    def sent_emails(self, result):
        return [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]

    def assert_no_attachment_claim(self, text):
        claims = (
            r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?) attached\b|"
            r"\b(?:please find|see|with the) (?:the )?(?:attached|enclosed)\b|"
            r"\b(?:deck|file|document|attachment) (?:is|was|has been) "
            r"(?:attached|included|sent|enclosed)\b"
        )
        self.assertNotRegex(text.lower(), claims)

    def test_pricing_deck_without_link(self):
        result = self.send_request("Email LEAD-50003 to send over our pricing deck.")
        reply = result["reply"]
        self.assertRegex(
            reply.lower(),
            r"cannot|can't|unable|does not support|not.*(?:attached|included)|without",
        )
        self.assert_no_attachment_claim(reply)
        emails = self.sent_emails(result)
        if emails:
            for email in emails:
                self.assertEqual(email["status"], "sent")
                self.assert_no_attachment_claim(email["body"])
                self.assertNotRegex(email["body"], r"https?://")
            self.assertRegex(
                reply.lower(),
                r"(?:not|without|no)\b.*(?:deck|document|attachment|attached|included)",
            )
        else:
            self.assertRegex(reply.lower(), r"link|url")
            self.assertNotRegex(reply.lower(), r"\b(?:i|we) sent\b")

    def test_ordinary_email(self):
        result = self.send_request(
            "Email LEAD-50003 to ask whether Tuesday works for a call."
        )
        emails = self.sent_emails(result)
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0]["status"], "sent")
        self.assertIn("Tuesday", emails[0]["body"])
        self.assert_no_attachment_claim(emails[0]["body"])
        self.assert_no_attachment_claim(result["reply"])
        self.assertRegex(result["reply"].lower(), r"sent|emailed")


if __name__ == "__main__":
    unittest.main()
