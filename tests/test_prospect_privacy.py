import copy
import importlib
import json
import unittest
from unittest.mock import Mock, patch

from langsmith import tracing_context

from gtm_agent import data_service

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-TEST"
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 90000000,
            "tech_stack": ["PostgreSQL"],
            "account_details": [{"industry": "Software"}],
            "engagement_history": [{"channel": "Demo"}],
            "enrichment_source": "Test provider",
            "billing_qualification": {"tax_id": "private-test-value"},
        }
        self.enterContext(patch.dict(data_service.PROSPECTS,
                                     {self.prospect_id: self.record}, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.enterContext(tracing_context(enabled=False))

    def test_contact_fields_only(self):
        for disqualified in (None, False, True):
            with self.subTest(disqualified=disqualified):
                if disqualified is not None:
                    self.record["disqualified"] = disqualified
                self.assertEqual(agent.get_prospect.invoke({"prospect_id": self.prospect_id}), {
                    "prospect": {
                        "prospect_id": self.prospect_id,
                        "name": self.record["name"],
                        "email": self.record["email"],
                        "disqualified": bool(disqualified),
                    },
                    "found": True,
                })

    def test_fresh_profile_preserves_enrichment_without_billing(self):
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        expected["prospect_id"] = self.prospect_id
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        self.assertIn("billing_qualification", self.record)

    def test_cached_profile_is_sanitized_before_traced_return(self):
        cached = {"prospect_id": self.prospect_id, **copy.deepcopy(self.record)}
        data_service._PROFILES[self.prospect_id] = cached
        expected = {k: v for k, v in cached.items() if k != "billing_qualification"}
        runs = []
        with tracing_context(enabled="local"):
            result = data_service.get_profile_from_db(
                self.prospect_id, langsmith_extra={"on_end": runs.append})
        self.assertEqual(result, {"prospect_profile": expected})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].outputs, result)
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
                         {"prospect_profile": expected, "found": True})

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"),
                         {"prospect_profile": None})

    def test_scoring_payload_only_contains_scoring_fields(self):
        profile = {"prospect_id": self.prospect_id, **copy.deepcopy(self.record)}
        profile["tech_stack"] = ["Outdated technology"]
        offering = data_service.OFFERINGS["OFFER-10001"]
        score = {"score": 70, "justification": "Test score"}
        llm = Mock()
        llm.invoke.return_value.model_dump.return_value = score
        with patch.object(agent, "_scoring_llm", llm):
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertEqual(result, score)
        user_message = llm.invoke.call_args.args[0][1]["content"]
        payload = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(payload, {
            "prospect_id": self.prospect_id,
            **{k: self.record[k] for k in (
                "name", "annual_revenue", "tech_stack", "account_details", "engagement_history")},
        })
        self.assertNotIn("private-test-value", user_message)
        self.assertEqual(profile["tech_stack"], ["Outdated technology"])

    def test_scoring_still_requires_valid_offering(self):
        for offering in (None, {}, {"description": "Incomplete"}):
            with self.subTest(offering=offering):
                result = agent.score_prospect.invoke({"prospect_profile": self.record,
                                                      "offering": offering})
                self.assertEqual(result, {"score": None,
                                          "error": "Cannot score without a valid offering."})


if __name__ == "__main__":
    unittest.main()
