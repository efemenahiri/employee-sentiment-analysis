"""Builds report/Final_Report.docx from outputs/*.csv and visualization/*.png.
Run from the project root after the notebook:  python tools/build_report.py"""
import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = lambda f: pd.read_csv(os.path.join(ROOT, "outputs", f))
V = lambda f: os.path.join(ROOT, "visualization", f)
lab = pd.read_csv(os.path.join(ROOT, "data", "test_labeled.csv"))
rank, risk, trig = O("overall_ranking.csv"), O("flight_risks.csv"), O("flight_risk_trigger_days.csv")
met = O("model_metrics.csv").set_index("Unnamed: 0"); coef = O("model_coefficients.csv")
share = lab["sentiment"].value_counts(); n = len(lab)

doc = Document()
doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10.5)

def table(df, widths=None):
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(r): cells[i].text = f"{v:.3f}" if isinstance(v, float) else str(v)
    doc.add_paragraph()
def fig(name, caption, w=6.3):
    doc.add_picture(V(name), width=Inches(w)); doc.add_paragraph(caption).italic = True
def bullets(items):
    for t in items: doc.add_paragraph(t, style="List Bullet")

doc.add_heading("Employee Sentiment & Engagement Analysis - Final Report", 0)
doc.add_paragraph("Dataset: test.csv (2,191 unlabelled employee e-mails, 10 senders, Jan 2010 - Dec 2011). "
                  "Code: notebooks/sentiment_analysis.ipynb (main file) with supporting modules in src/.")

doc.add_heading("1. Summary", 1)
top = rank.head(3); low = rank.tail(3).iloc[::-1]
bullets([
 f"Sentiment labels: {share['Neutral']:,} Neutral ({share['Neutral']/n:.1%}), {share['Positive']:,} Positive "
 f"({share['Positive']/n:.1%}), {share['Negative']:,} Negative ({share['Negative']/n:.1%}).",
 "Top three positive employees (whole-period total of monthly scores): " + ", ".join(f"{r['name']} ({r.total_score})" for _, r in top.iterrows()) + ".",
 "Three lowest-scoring employees: " + ", ".join(f"{r['name']} ({r.total_score})" for _, r in low.iterrows()) + ". All ten totals are positive, so 'lowest' means least positive.",
 "Flight risks (4+ negative e-mails in any rolling 30 days): " + ", ".join(risk["name"]) + ". The list is threshold-sensitive (see section 5).",
 f"Regression: monthly score is predicted moderately well (test R-squared {met.loc['r2_test','A: monthly_score']:.2f}) mainly because it grows with message count; "
 f"tone per message is not predictable from structural features (test R-squared {met.loc['r2_test','B: mean_sentiment']:.2f}).",
])

doc.add_heading("2. Approach and methodology", 1)
doc.add_paragraph("Labelling (Task 1). No pretrained model or internet access was available, so a VADER-style, rule-based scorer was written "
 "from scratch (src/sentiment.py) with a hand-curated business-email lexicon (about 390 words, valence -4..+4). Negations flip and dampen a word, "
 "boosters and dampeners adjust it, valences are summed and squashed with compound = s / sqrt(s^2 + 15). A message is Positive if compound >= 0.25, "
 "Negative if <= -0.25, otherwise Neutral. The subject is prepended to the body. No LLM was used; the method is deterministic and reproducible.")
doc.add_paragraph("Cleaning (src/preprocessing.py). Quoted-printable codes, URLs, e-mail addresses, quoted/forwarded history, legal confidentiality footers and "
 "closing salutations ('Best regards') are removed so only the sender's own words are scored. Dates are parsed as month/day/year (verified: no first field exceeds 12).")
doc.add_paragraph("Validation. A hand-check of 40 random messages judged 33 of 40 labels reasonable; this is a small subjective check, not a measured accuracy, "
 "because the data has no ground-truth labels. After reading the messages behind the first flight-risk results, three artefacts were corrected "
 "(legal footers, 'Best regards', and the words 'risk' and 'outage'). These changes were made after seeing results, so the labels are tuned by inspection.")

doc.add_heading("3. Exploratory data analysis (Task 2)", 1)
bullets([
 "2,191 rows x 4 columns (Subject, body, date, from); no formally missing values; the file is tab-separated despite the .csv extension.",
 "Data quality: 243 messages have no subject, 32 have no usable text (scored Neutral), 2 bodies hit the 999-character export limit (truncation).",
 "The date column looks synthetic: every month has exactly 91-92 messages, and 576 bodies mention 1999-2002 while none mention 2010-2011. Time trends are therefore trends over the dates in the file, not necessarily real chronology.",
 "About 49% of rows (1,074) share subject and body with another row, mostly the same mail under different senders; only 82 are same-sender repeats.",
 "Neutral dominates; Positive/Negative messages are about twice as long (median 44-48 words vs 21 for Neutral), partly because longer text has more chances to contain a sentiment word.",
 "Mean sentiment per message drifts up by about +0.0046 per month (r = 0.63, p = 0.001).",
 "Differences between employees are modest: positive share 22% (Rhonda Denton) to 35% (Eric Bass); negative share 6% (John Arnold) to 9% (Kayne Coulter).",
])
fig("01_sentiment_distribution.png", "Figure 1 - Sentiment label distribution.")
fig("02_monthly_volume_and_trend.png", "Figure 2 - Monthly volume and average sentiment trend.")
fig("03_sentiment_by_employee.png", "Figure 3 - Sentiment mix by employee.")
fig("04_length_by_sentiment.png", "Figure 4 - Message length by sentiment label.")

doc.add_heading("4. Employee scoring and ranking (Tasks 3 and 4)", 1)
doc.add_paragraph("Scoring: Positive = +1, Negative = -1, Neutral = 0, summed per employee per calendar month (group by month and employee). "
 "The score resets each month because a message belongs only to the month it was sent in. Checks: totals reconcile with message-level scores and one "
 "employee-month was recomputed by hand. All 240 employee-months exist; 10 of them have a negative score.")
doc.add_paragraph("Ranking: for every month, the top three highest scores (descending, ties alphabetical) and the three lowest scores (most negative first, ties alphabetical). "
 "Ties at the cut-off are decided by name, so monthly lists are noisy; the full month-by-month lists are in outputs/monthly_top3_positive.csv and monthly_top3_negative.csv. "
 "Whole-period totals:")
table(rank[["name", "total_score", "n_messages", "n_positive", "n_negative", "score_per_message"]].rename(columns={"name": "employee"}))
fig("05_employee_month_heatmap.png", "Figure 5 - Monthly score per employee.")
fig("06_overall_ranking.png", "Figure 6 - Whole-period ranking.")
fig("07_monthly_top3_appearances.png", "Figure 7 - Months each employee appears in the monthly top-3 lists.")

doc.add_heading("5. Flight-risk identification (Task 5)", 1)
doc.add_paragraph("Rule: an employee is a flight risk if they sent 4 or more Negative e-mails within any rolling 30-day window (the day plus the previous 29), "
 "irrespective of score and of month boundaries. Several e-mails on the same day all count. Implementation: pandas rolling('30D') over each employee's negative messages, "
 "cross-checked against an independent brute-force implementation and covered by unit tests (boundary 29/30 days, month crossing, same-day, ignoring other sentiments).")
table(risk[["name", "first_flagged", "last_flagged", "max_negatives_in_30d", "months_affected"]].rename(columns={"name": "employee"}))
doc.add_paragraph("Evidence: the four negative e-mails behind the flag are a 'Transition Issues' note, a 'regret to inform you ... call canceled' notice, "
 "fantasy-football banter ('your running backs suck') and a long deal e-mail with a few cautionary words - mixed, and worth reading before drawing conclusions.")
doc.add_paragraph("Robustness: removing same-sender duplicates changes nothing. The label threshold matters a lot: 7 employees are flagged at +-0.15, 1 at +-0.25 (used), "
 "none at +-0.35. An earlier lexicon version, before the footer/sign-off/'risk' fixes, flagged four employees. Treat the flag as a prompt for a conversation, not as evidence of intent to leave.")
fig("08_flight_risk_timeline.png", "Figure 8 - Rolling 30-day negative count per employee (threshold 4).")

doc.add_heading("6. Predictive modelling (Task 6)", 1)
doc.add_paragraph("Data: one row per employee-month (240 rows). Features (deliberately not sentiment words, to avoid label leakage): n_messages, avg_word_count, avg_char_length, "
 "avg_subject_words, share_questions, share_exclaims, month_index. Two targets: A = monthly_score (the Task 3 score) and B = mean_sentiment (score per message). "
 "80/20 random split (seed 42); scikit-learn LinearRegression; metrics on the held-out 20% compared with a predict-the-training-mean baseline, plus 5-fold CV on the training set. "
 "Coefficient p-values are from OLS t-tests on the training data.")
table(met.reset_index().rename(columns={"Unnamed: 0": "metric"}))
doc.add_paragraph("Coefficients (training data):")
table(coef)
bullets([
 "Model A: n_messages is the dominant predictor (about +0.17 points per extra message, p < 0.001) - largely arithmetic, since the score is a sum. month_index is also significant (p = 0.028), echoing the upward drift. Test R-squared 0.48 vs CV R-squared 0.28, so the test figure is partly luck of the split.",
 "Model B: essentially no predictive power (test R-squared 0.05, CV R-squared negative). Structural features do not forecast tone once volume is controlled for.",
 "Residuals for Model A are centred near zero but right-skewed (a few very high-score months are under-predicted).",
])
fig("09_model_actual_vs_predicted.png", "Figure 9 - Actual vs predicted (test set) and Model A residuals.")
fig("10_model_coefficients.png", "Figure 10 - Standardised coefficients.")

doc.add_heading("7. Limitations", 1)
bullets([
 "Lexicon-based labels: no sarcasm or context understanding; accuracy was not measured against ground truth (author hand-check: 33/40 reasonable).",
 "Labels were tuned after inspecting flagged messages; flight-risk results are threshold-sensitive and marginal (exactly 4 negatives).",
 "Synthetic-looking dates, truncated bodies and many duplicated mails limit time-based conclusions.",
 "240 observations and a lexicon-generated regression target; linear model on small bounded integers.",
])
doc.add_heading("8. Recommendations", 1)
bullets([
 "Validate the labels on a human-labelled sample (or with an LLM) before any HR decision; re-run the notebook with the better labels - all downstream steps recompute.",
 "Use the flight-risk output only as a prompt for a manager conversation, after reading the evidence messages.",
 "Use score per message alongside the total when comparing people, because totals scale with volume.",
 "Verify the date field with the data owner before relying on trends.",
])
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
doc.save(os.path.join(ROOT, "report", "Final_Report.docx")); print("saved report")
