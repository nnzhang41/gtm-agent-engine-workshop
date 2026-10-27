import copy
import json
import unittest
from unittest.mock import patch

with patch.dict("os.environ", OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    from gtm_agent import data_service, gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        records = copy.deepcopy(data_service.PROSPECTS)
        self.enterContext(patch.dict(data_service.PROSPECTS, records, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.enterContext(patch.dict("os.environ", LANGSMITH_TRACING="false"))

    def test_update_persists_and_refreshes_profile_for_scoring(self):
        prospect_id = "LEAD-39002"
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        stale_profile = original["prospect_profile"]
        self.assertNotIn("Terraform", stale_profile["tech_stack"])

        result = data_service.update_prospect_info(prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(prospect_id))
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

        offering = data_service.get_offering("OFFER-10004")
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            gtm_agent.score_prospect.invoke({
                "prospect_profile": stale_profile,
                "offering": offering,
            })
        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertIn("Terraform", scored_profile["tech_stack"])
        self.assertFalse(set(offering["required_tech_stack"]) - set(scored_profile["tech_stack"]))

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_missing_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("missing", "Terraform"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("missing"))
