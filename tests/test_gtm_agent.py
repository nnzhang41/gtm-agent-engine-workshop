import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    def test_application_tools_through_run_agent(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
            agent = importlib.import_module("gtm_agent.gtm_agent")
            rep = agent.REP_IDS[0]
            prospect_id = next(iter(agent.data_service.PROSPECTS))
            prospect = agent.data_service.get_prospect_record(prospect_id)
            offering_id = next(iter(agent.data_service.OFFERINGS))
            offering = agent.data_service.get_offering(offering_id)
            score = agent.ProspectScore(
                score=75, justification="Fixture score",
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=80, tech_stack_match=70, segment_fit=75,
                ),
            )
            calls = [
                ("get_current_rep", {}),
                ("lookup_offering", {"offering_id": offering_id}),
                ("build_prospect_profile", {"prospect_id": prospect_id}),
                ("get_prospect", {"prospect_id": prospect_id}),
                ("send_prospect_email", {
                    "prospect": prospect, "subject": "Hello", "body": "Welcome",
                }),
                ("score_prospect", {
                    "prospect_profile": {"prospect_id": prospect_id}, "offering": offering,
                }),
                ("update_prospect_info", {
                    "prospect_id": prospect_id, "technology": "Test technology",
                }),
            ]
            requests = []

            def respond(model, messages, **kwargs):
                requests.append((messages, kwargs))
                index = len(requests) - 1
                if index < len(calls):
                    name, args = calls[index]
                    message = AIMessage(content="", tool_calls=[{
                        "name": name, "args": args, "id": str(index), "type": "tool_call",
                    }])
                else:
                    message = AIMessage(content="Completed all seven tools.")
                return ChatResult(generations=[ChatGeneration(message=message)])

            with patch.object(ChatOpenAI, "_generate", respond), \
                    patch.object(agent, "_scoring_llm", Mock(invoke=Mock(return_value=score))), \
                    patch.dict(agent.data_service._PROFILES, clear=True):
                result = agent.run_agent(
                    "Email and score this prospect", user_id=rep["rep_id"],
                    thread_id="test-thread", environment="test",
                )

            self.assertEqual(result["reply"], "Completed all seven tools.")
            self.assertTrue(result["run_id"])
            results = {
                message.name: json.loads(message.content)
                for message in result["messages"] if isinstance(message, ToolMessage)
            }
            self.assertEqual(set(results), {name for name, _ in calls})
            self.assertEqual(results["get_current_rep"], {"rep": rep, "found": True})
            self.assertEqual(results["lookup_offering"]["offering"], offering)
            self.assertEqual(results["build_prospect_profile"]["prospect_profile"]["prospect_id"], prospect_id)
            self.assertEqual(results["get_prospect"]["prospect"]["email"], prospect["email"])
            self.assertEqual(results["send_prospect_email"]["status"], "sent")
            self.assertEqual(results["send_prospect_email"]["from"], rep["email"])
            self.assertEqual(results["send_prospect_email"]["to"], prospect["email"])
            self.assertEqual(results["score_prospect"], score.model_dump())
            self.assertTrue(results["update_prospect_info"]["updated"])
            self.assertIn("Test technology", results["update_prospect_info"]["tech_stack"])
            for messages, kwargs in requests:
                self.assertEqual(
                    {tool["function"]["name"] for tool in kwargs["tools"]},
                    {name for name, _ in calls},
                )
                for tool in kwargs["tools"]:
                    self.assertNotIn("runtime", tool["function"]["parameters"]["properties"])
                self.assertEqual(
                    [message.content for message in messages if message.type == "system"],
                    [agent.SYSTEM_PROMPT],
                )


if __name__ == "__main__":
    unittest.main()
