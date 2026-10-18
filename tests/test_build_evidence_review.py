import unittest
from src.evaluation.build_evidence_review import build_review_rows

class EvidenceReviewTests(unittest.TestCase):
    def test_cross_product_and_blank_reviewer_fields(self):
        pilot = {"questions": [{"sample_id":"s1", "question_type":"comparison_query", "query":"Q?", "answer":"Yes"}]}
        ann = {"annotations": [{
            "sample_id":"s1", "question_type":"comparison_query", "query":"Q?", "gold_answer":"Yes",
            "required_claims":[{"claim_id":"C1","claim":"claim one"},{"claim_id":"C2","claim":"claim two"}],
            "evidence_items":[{"evidence_id":"E1","title":"T1","fact":"F1"},{"evidence_id":"E2","title":"T2","fact":"F2"}],
            "evidence_support":[{"claim_id":"C1","evidence_id":"E1","label":"supports","rationale":"draft"}]
        }]}
        rows = build_review_rows(pilot, ann)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["draft_support_status"], "supports")
        self.assertEqual(rows[1]["draft_support_status"], "not_annotated")
        self.assertTrue(all(row["reviewer_decision"] == "" for row in rows))
        self.assertTrue(all(row["review_answer_entailment"] == "" for row in rows))

    def test_samples_without_draft_claims_still_appear(self):
        pilot = {"questions": [{"sample_id":"s2", "question_type":"temporal_query", "query":"Q2?", "answer":"No"}]}
        ann = {"annotations": [{"sample_id":"s2", "query":"Q2?", "gold_answer":"No",
            "required_claims": [], "evidence_items":[{"evidence_id":"E1","fact":"F1"},{"evidence_id":"E2","fact":"F2"}], "evidence_support": []}]}
        rows = build_review_rows(pilot, ann)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["sample_id"] for r in rows}, {"s2"})
        self.assertTrue(all(r["draft_support_status"] == "no_draft_claims" for r in rows))

if __name__ == "__main__":
    unittest.main()
