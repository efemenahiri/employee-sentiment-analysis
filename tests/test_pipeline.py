"""Unit tests: python -m unittest discover -s tests -v   (from the project root)"""
import os, sys, unittest
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_body, clean_subject, prepare, load_raw
from src.sentiment import label_text, score_text
from src.analysis import (add_message_score, monthly_scores, rank_monthly, flight_risk_windows,
                          flight_risk_bruteforce, flight_risk_summary)

def mk(rows):
    df = pd.DataFrame(rows, columns=["employee", "date", "sentiment"])
    df["date"] = pd.to_datetime(df["date"]); df["month"] = df["date"].dt.to_period("M").astype(str)
    return add_message_score(df)

class TestCleaning(unittest.TestCase):
    def test_quoted_printable_and_forward(self):
        t = clean_body("Hello=20there =\nfriend\n-----Original Message-----\nFrom: x\nTERRIBLE")
        self.assertEqual(t, "Hello there friend"); 
    def test_disclaimer_and_signoff_removed(self):
        t = clean_body("See attached. Best regards, Tom ------ The information transmitted is intended only for the person; if received in error delete it")
        self.assertNotIn("error", t); self.assertNotIn("regards", t.lower())
        self.assertEqual(label_text(clean_body("Hi Sam, the file is attached. Best regards")), "Neutral")
    def test_subject(self):
        self.assertEqual(clean_subject("RE: FW: Budget"), "Budget"); self.assertEqual(clean_subject("(No Subject)"), "")

class TestSentiment(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(label_text("Congratulations, this is excellent news!"), "Positive")
        self.assertEqual(label_text("This is a terrible disaster and I am furious"), "Negative")
        self.assertEqual(label_text("Meeting moved to 3pm in room 4"), "Neutral")
        self.assertEqual(label_text(""), "Neutral")
    def test_domain_terms_are_neutral(self):
        self.assertEqual(label_text("RAC risk reporting for the risk management group"), "Neutral")
        self.assertEqual(label_text("Outlook outage notice for Monday"), "Neutral")
    def test_negation(self):
        self.assertLess(score_text("this is not good"), score_text("this is good"))
        self.assertLess(score_text("not good"), 0.25)

class TestScoring(unittest.TestCase):
    def test_monthly_reset(self):
        d = mk([("a", "2010-01-31", "Negative"), ("a", "2010-02-01", "Positive"),
                ("a", "2010-02-02", "Positive"), ("a", "2010-02-03", "Neutral")])
        s = monthly_scores(d).set_index("month")["monthly_score"]
        self.assertEqual(s["2010-01"], -1); self.assertEqual(s["2010-02"], 2)
    def test_ranking_ties_alphabetical(self):
        d = mk([(e, "2010-03-01", "Positive") for e in "dcba"] + [("z", "2010-03-02", "Negative")])
        pos, neg = rank_monthly(monthly_scores(d))
        self.assertEqual(list(pos.employee), ["a", "b", "c"]); self.assertEqual(neg.employee.iloc[0], "z")

class TestFlightRisk(unittest.TestCase):
    def test_boundaries(self):
        # 4 negatives, first and last 29 days apart -> flagged
        a = mk([("a", d, "Negative") for d in ["2010-01-01", "2010-01-10", "2010-01-20", "2010-01-30"]])
        self.assertEqual(set(flight_risk_summary(a).employee), {"a"})
        # first and last 30 days apart -> NOT flagged
        b = mk([("b", d, "Negative") for d in ["2010-01-01", "2010-01-10", "2010-01-20", "2010-01-31"]])
        self.assertTrue(flight_risk_summary(b).empty)
    def test_spans_month_boundary(self):
        c = mk([("c", d, "Negative") for d in ["2010-01-28", "2010-01-30", "2010-02-02", "2010-02-05"]])
        self.assertEqual(set(flight_risk_summary(c).employee), {"c"})   # 2 per month, still flagged
    def test_same_day_and_ignores_other_sentiment(self):
        d = mk([("d", "2010-05-01", "Negative")] * 4)
        self.assertEqual(set(flight_risk_summary(d).employee), {"d"})
        e = mk([("e", "2010-05-01", "Negative")] * 3 + [("e", "2010-05-02", "Neutral")])
        self.assertTrue(flight_risk_summary(e).empty)
    def test_matches_bruteforce_on_real_data(self):
        from src.sentiment import label_dataframe
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test.csv")
        df = add_message_score(label_dataframe(prepare(load_raw(p))))
        self.assertEqual(set(flight_risk_summary(df).employee), flight_risk_bruteforce(df))

if __name__ == "__main__":
    unittest.main()
