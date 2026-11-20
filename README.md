# Employee Sentiment & Engagement Analysis

Analysis of 2,191 unlabelled employee e-mails (`data/test.csv`, 10 senders, Jan 2010 - Dec 2011): sentiment labelling, EDA,
monthly employee scores, rankings, flight-risk detection and a linear-regression model.

**Main file:** `notebooks/sentiment_analysis.ipynb` (all steps, with titles, commentary and observations).
**Full write-up:** `report/Final_Report.docx`. **Charts:** `visualization/`. **Result tables:** `outputs/`.

## Results at a glance

| | |
|---|---|
| Labels | 1,382 Neutral (63.1%), 646 Positive (29.5%), 163 Negative (7.4%) |
| **Top 3 positive** (sum of monthly scores, 24 months) | 1. Lydia Delgado (66), 2. Eric Bass (60), 3. John Arnold (59) |
| **Top 3 negative** (lowest sum) | 1. Rhonda Denton (25), 2. Kayne Coulter (37), 3. Patti Thompson (38) |
| **Flight risks** (4+ negative e-mails in any rolling 30 days) | Bobette Riner (4 negatives, 30 Dec 2010 - 28 Jan 2011) |
| Regression | Monthly score: test R² 0.48, RMSE 1.48 (baseline 2.06), 5-fold CV R² 0.28. Tone per message: test R² 0.05 (not predictable) |

Per-month top-3 positive and negative lists are in `outputs/monthly_top3_positive.csv` and `outputs/monthly_top3_negative.csv`
(sorted by score, then alphabetically). Note that all ten overall totals are positive, so "top negative" means *least positive*.

## Key insights and recommendations

- Tone is mostly neutral and mildly positive, with a small upward drift (+0.0046 mean score per message per month, p = 0.001).
- **Treat the flight-risk list with caution.** TextBlob labels would flag 9 of 10 employees (see above). Bobette Riner reached exactly 4 negatives, and the evidence is mixed (two mildly negative business notes, one fantasy-football joke, one cautionary deal e-mail). The list depends on the label threshold: 7 employees at ±0.15, 1 at ±0.25 (used), none at ±0.35.
- Totals scale with message volume; compare people on **score per message** too (Eric Bass highest, Rhonda Denton lowest).
- The regression shows volume (not writing style) explains the monthly score; no structural feature predicts tone.
- Before any HR use: validate labels against human labels or an LLM, and verify the `date` column (it looks synthetic: exactly 91-92 messages every month, bodies refer to 1999-2002).

## TextBlob cross-check (notebook section 1.1)

The assessment form asks for TextBlob. The headline results above use my own lexicon labels, and the notebook also runs
**TextBlob** (polarity, cut-off ±0.05) on the same cleaned text and compares the two. Results from running it:

| | Lexicon (used for headline results) | TextBlob |
|---|---|---|
| Positive | 29.5% | 53.4% |
| Neutral | 63.1% | 34.3% |
| Negative | 7.4% | 12.3% |

- The two labelers **agree on 56.0%** of messages (1,228 of 2,191). The biggest gap: 578 messages the lexicon calls Neutral are Positive under TextBlob, and 185 are Negative.
- If TextBlob labels were used, the flight-risk rule would flag **9 of the 10 employees** (all except Kayne Coulter), versus 1 (Bobette Riner) with the lexicon labels. So the flight-risk list depends heavily on which labeler and cut-off is used, and neither should be treated as ground truth.
- TextBlob is a general-purpose tool and its ±0.05 cut-off is very sensitive, so mildly worded business messages often get a small non-zero polarity. I did not use TextBlob for the rankings or regression; per-message TextBlob results are in `outputs/textblob_labels.csv`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb     # then Run All
```
No API keys are needed (see `.env.example`; `DATA_PATH` optionally overrides the dataset location).

Command-line alternatives (run from the project root):
```bash
python -m unittest discover -s tests -v      # 12 unit tests (cleaning, labelling, scoring, ranking, flight risk)
python tools/build_notebook.py               # re-runs every step and rewrites the executed notebook, charts and outputs/
python tools/build_report.py                 # rebuilds report/Final_Report.docx from outputs/
```

## Project structure
```
data/test.csv            raw dataset (tab-separated despite the .csv name)
data/test_labeled.csv    dataset + sentiment_score, sentiment, message_score columns
src/preprocessing.py     loading, cleaning, date parsing
src/sentiment.py         rule-based sentiment lexicon scorer
src/analysis.py          monthly scores, ranking, flight risk, regression
notebooks/               main notebook (executed, with outputs)
tests/                   unit tests
tools/                   notebook builder/executor and report builder
visualization/           PNG charts      outputs/   CSV result tables      report/   Final_Report.docx
```

## Methodology (short)

1. **Labelling.** No pretrained model or internet was available, so a VADER-style lexicon scorer was written from scratch (~390 business-email words, negation/booster handling, `compound = s/√(s²+15)`). Label: ≥ +0.25 Positive, ≤ −0.25 Negative, else Neutral. Cleaning removes quoted history, URLs, quoted-printable codes, legal footers and "Best regards". No LLM was used (TextBlob is run as a cross-check, see above).
2. **Scoring.** Positive +1, Negative −1, Neutral 0, summed per employee per calendar month (resets monthly by construction).
3. **Ranking.** Per month: top 3 highest / lowest scores, sorted by score then alphabetically. Whole-period totals give the headline lists.
4. **Flight risk.** ≥ 4 Negative e-mails inside any rolling 30-day window (the day plus the previous 29), independent of month boundaries and score; cross-checked with a brute-force implementation.
5. **Regression.** Employee-month table (240 rows); features: message count, average word count, average length, subject length, share of questions / exclamations, month index (no sentiment words, to avoid leakage). 80/20 split, scikit-learn `LinearRegression`, OLS p-values, 5-fold CV, compared with a mean baseline.

## Limitations (please read)

- Labels come from a lexicon, not a trained model or human labels. A hand-check of 40 random messages judged **33 of 40** reasonable; that is a small subjective check, not a measured accuracy.
- After reading the messages behind the first flight-risk results, legal footers, "Best regards", and the words "risk"/"outage" were excluded from scoring. The final labels were therefore tuned by inspection; an earlier version flagged four employees.
- Many rows are duplicated across senders (49% share subject and body with another row), bodies are truncated at 999 characters, and dates look synthetic, so time-based conclusions are weak.
