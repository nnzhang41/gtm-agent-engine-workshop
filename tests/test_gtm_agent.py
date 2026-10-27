import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    seen_messages: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.seen_messages.append(messages)
        return super()._generate(messages, **kwargs)


class EmailTests(unittest.TestCase):
    def setUp(self):
        self.rep = agent.REP_IDS[0]
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.prospect = agent.data_service.get_prospect_record(self.prospect_id)
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})

    def test_send_resolves_recipient_and_sender(self):
        result = agent.send_prospect_email.func(
            self.prospect_id, "Hello", "Let's talk", self.runtime
        )
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["to_name"], self.prospect["name"])
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["from_name"], self.rep["name"])
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Let's talk")

    def test_unknown_prospect_does_not_send(self):
        with patch.object(agent.uuid, "uuid4") as message_id:
            result = agent.send_prospect_email.func("unknown", "Hello", "Body", self.runtime)
        self.assertEqual(result, {"status": "failed", "error": "Prospect not found."})
        message_id.assert_not_called()

    def test_missing_email_does_not_send(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"name": "Lead"}):
            result = agent.send_prospect_email.func(self.prospect_id, "Hello", "Body", self.runtime)
        self.assertEqual(result, {"status": "failed", "error": "Prospect record has no email address."})

    def replay(self, request, tool_name, args):
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": "call-1"}]),
            AIMessage(content="Done"),
        ])
        graph = agent.create_deep_agent(
            model=model,
            tools=[agent.send_prospect_email, agent.get_prospect, agent.get_current_rep],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        with patch.object(agent, "gtm_agent", graph), patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent.run_agent(request, user_id=self.rep["rep_id"])
        results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in results], [tool_name])
        return json.loads(results[0].content), model

    def test_email_replay_needs_only_send_tool(self):
        body = f"Let's talk. Regards, {self.rep['name']}"
        result, model = self.replay(
            f"Email {self.prospect_id} to arrange a call.",
            "send_prospect_email",
            {"prospect_id": self.prospect_id, "subject": "A call", "body": body},
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["body"], body)
        system_content = "\n".join(
            message.text for message in model.seen_messages[0] if message.type == "system"
        )
        self.assertIn(self.rep["name"], system_content)
        self.assertIn(self.rep["email"], system_content)
        self.assertEqual(len(model.seen_messages), 2)

    def test_contact_details_lookup_remains_available(self):
        result, _ = self.replay(
            f"What are the contact details for {self.prospect_id}?",
            "get_prospect", {"prospect_id": self.prospect_id},
        )
        self.assertTrue(result["found"])
        self.assertEqual(result["prospect"]["email"], self.prospect["email"])

    def test_rep_identity_lookup_remains_available(self):
        result, _ = self.replay("Who is the signed-in rep?", "get_current_rep", {})
        self.assertTrue(result["found"])
        self.assertEqual(result["rep"], self.rep)


if __name__ == "__main__":
    unittest.main()
