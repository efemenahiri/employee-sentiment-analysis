"""Builds notebooks/sentiment_analysis.ipynb and EXECUTES it without needing Jupyter.

Jupyter/nbconvert were not installed in the environment this project was built in, so
this script contains (a) the notebook cells and (b) a tiny executor that runs each code
cell in one shared namespace, captures stdout, DataFrame displays and matplotlib figures,
and stores them as normal .ipynb outputs.  In your own environment you can simply open
the notebook in Jupyter / VS Code and "Run All" - the cells are ordinary Python.

Usage (from the project root):  python tools/build_notebook.py
"""
import ast, base64, contextlib, io, json, os, sys, textwrap, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebooks", "sentiment_analysis.ipynb")

CELLS = []
def md(t): CELLS.append(("markdown", textwrap.dedent(t).strip("\n")))
def code(t): CELLS.append(("code", textwrap.dedent(t).strip("\n")))

# ============================================================================ CELLS
md("""
# Employee Sentiment & Engagement Analysis

**Goal.** Starting from an *unlabelled* file of employee e-mails (`data/test.csv`) we
1. label every message Positive / Negative / Neutral,
2. explore the data (EDA),
3. compute a monthly sentiment score per employee,
4. rank employees (top-3 positive / top-3 negative),
5. flag *flight risks* (>= 4 negative e-mails inside any rolling 30-day window), and
6. fit a linear-regression model of sentiment trends.

**How to read this notebook.** Every section has a title, a short statement of method,
the code, and an **Observations** note written after looking at the output.
Reusable logic lives in `src/` (`preprocessing.py`, `sentiment.py`, `analysis.py`) and is
unit-tested in `tests/`; this notebook is the narrative that ties it together.

| Section | Task |
|---|---|
| 0 | Setup & data loading |
| 1 | Sentiment labelling |
| 2 | Exploratory data analysis |
| 3 | Monthly employee scores |
| 4 | Employee ranking |
| 5 | Flight-risk identification |
| 6 | Predictive modelling (linear regression) |
| 7 | Summary & exports |
""")

md("## 0. Setup & data loading")
code("""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg") if os.environ.get("MPLBACKEND") is None and not hasattr(sys, "ps1") and "ipykernel" not in sys.modules else None
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

# make `src` importable whether the notebook is run from notebooks/ or the project root
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, ROOT)
from src.preprocessing import load_raw, prepare
from src.sentiment import (label_dataframe, label_from_score, explain_text,
                           POS_THRESHOLD, NEG_THRESHOLD)
from src.analysis import (add_message_score, monthly_scores, rank_monthly, overall_ranking,
                          flight_risk_windows, flight_risk_summary, flight_risk_bruteforce,
                          build_model_table, fit_linear_model)

DATA = os.environ.get("DATA_PATH") or os.path.join(ROOT, "data", "test.csv")   # optional override, see .env.example
VIZ = os.path.join(ROOT, "visualization"); OUT = os.path.join(ROOT, "outputs")
os.makedirs(VIZ, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# ---- plotting style: one consistent, colour-blind-safe set (blue / grey / orange) ----
C = {"Positive": "#2a78d6", "Neutral": "#a8a79f", "Negative": "#eb6834",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3de", "surface": "#fcfcfb"}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "legend.frameon": False, "figure.dpi": 100,
})
def save(fig, name):
    fig.savefig(os.path.join(VIZ, name), dpi=150, bbox_inches="tight"); plt.show()
def short(e): return e.split("@")[0].replace(".", " ").title()
ORDER = ["Positive", "Neutral", "Negative"]
""")
code("""
raw = load_raw(DATA)
print("Raw file shape:", raw.shape)
print("Columns:", list(raw.columns))
raw.head(3)
""")
md("""
**Observations.** `test.csv` is really *tab-separated* (despite the extension) and message
bodies contain quoted line-breaks, so the loader auto-detects the delimiter and uses a quote
character. Four columns: `Subject`, `body`, `date` (month/day/year), `from` (the sender - the
"employee"). Bodies also contain mail-transport artefacts (`=20`, `=\\n` quoted-printable codes)
and quoted earlier messages, which would pollute sentiment scoring - the cleaning step below removes them.
""")
code("""
df = prepare(raw)          # parse dates, clean subject/body, build the text that will be scored
df[["employee", "date", "month", "subject_clean", "body_clean"]].head(5)
""")

md("""
## 1. Sentiment labelling (Task 1)

**Method - a transparent, reproducible lexicon scorer** (`src/sentiment.py`).

* *Why not an LLM?* The brief prefers an LLM "or any suitable NLP technique". This project was
  built in an offline environment with no pretrained models, so it uses a VADER-style
  lexicon approach written from scratch. It is deterministic (same input -> same label), free
  to run, and every decision can be inspected. No LLM was used; see the limitations in
  Section 7 and the README for how an LLM could be swapped in and used to validate these labels.
* *Cleaning first.* Quoted-printable codes, URLs, e-mail addresses, quoted/forwarded history
  (`-----Original Message-----`, `>` lines) are removed so that only the sender's own words
  are scored. The subject is prepended to the body.
* *Scoring.* Each word has a valence (-4..+4). Negations ("not good") flip & dampen, boosters
  ("very") amplify. Valences are summed and squashed: `compound = s / sqrt(s^2 + 15)` in [-1, 1].
* *Labels.* `compound >= +0.25` -> **Positive**, `<= -0.25` -> **Negative**, otherwise **Neutral**.
  The band is wider than VADER's default (+-0.05) because business e-mail is full of politeness
  ("thanks", "please") that should not make a scheduling note "Positive".
* *Domain adjustments (found by error analysis).* After reading the messages behind the first
  flight-risk results, three artefacts were fixed: legal confidentiality footers (which contain the word
  "error") and closing salutations ("Best regards" -> the positive word "best") are stripped in
  `preprocessing.py`, and "risk" / "outage" (department names and routine IT notices in this corpus) carry no
  sentiment. These changes were made *after* seeing the first results, so the final labels are tuned by
  inspection, not blind - another reason to treat the flags as indicative.
""")
code("""
df = label_dataframe(df)    # adds sentiment_score (compound) and sentiment (label)
counts = df["sentiment"].value_counts().reindex(ORDER)
pd.DataFrame({"messages": counts, "share_%": (counts / len(df) * 100).round(1)})
""")
code("""
# Spot-check: 4 random messages per label, with the lexicon words that drove each decision
pd.set_option("display.max_colwidth", 140, "display.width", 200)
rows = []
for lab in ORDER:
    for _, r in df[(df.sentiment == lab) & (df.text.str.len() > 0)].sample(4, random_state=11).iterrows():
        rows.append({"label": lab, "score": round(r.sentiment_score, 2),
                     "text": r.text[:130], "drivers": ", ".join(f"{w}({v:+.1f})" for w, v in explain_text(r.text)[:5])})
pd.DataFrame(rows)
""")
code("""
# Sensitivity: how much do the label shares depend on the +-threshold?
sens = {}
for t in (0.15, 0.25, 0.35):
    lab = df["sentiment_score"].map(lambda s: label_from_score(s, t, -t))
    sens[f"+-{t}"] = (lab.value_counts(normalize=True).reindex(ORDER) * 100).round(1)
pd.DataFrame(sens).rename_axis("share % at threshold")
""")
md("""
**Observations.**
* Neutral dominates (about **63%**), then Positive (**30%**), and Negative is a small minority
  (**7%**, 163 messages). That is plausible for routine corporate e-mail (scheduling, forwarding,
  data requests) with occasional congratulations/pleasantries and rarer complaints.
* The spot-check shows labels driven by sensible words (e.g. *congratulations, appreciate* ->
  Positive; *sorry, wrong, problem* -> Negative). Known weaknesses: no sarcasm detection, and
  words with trade meaning ("loss matrix", "cancellation notice") can still mis-fire.
  A hand-check of 40 random messages (seed 7, reading the first ~260 characters of each) judged
  **33 of 40** labels reasonable. The 7 doubtful ones were mostly neutral-in-spirit mails given a
  Positive/Negative label on one or two words (job enquiries, an orientation notice, a
  "customer requested cancel" notice, a joke). This is a small, subjective check by the author,
  **not** a measured accuracy (there are no ground-truth labels).
* The sensitivity table shows the Negative share moves between ~5% and ~10% as the threshold goes
  from 0.35 to 0.15, so **Negative counts (and therefore flight-risk results) are threshold-dependent**.
  We keep +-0.25 as the documented, fixed choice.
""")

md("## 2. Exploratory data analysis (Task 2)")
md("### 2.1 Structure, types, missing values and data quality")
code("""
print("Records:", len(df), "| Columns in raw file:", raw.shape[1])
print("Date range:", df.date.min().date(), "->", df.date.max().date(), "| months:", df.month.nunique())
print("Employees (senders):", df.employee.nunique())
print()
print(raw.dtypes.rename("dtype").to_frame().join(raw.isna().sum().rename("missing")))
quality = pd.Series({
    "rows with '(No Subject)'/empty subject": int((df.subject_clean == "").sum()),
    "rows whose body is empty after cleaning": int((df.body_clean == "").sum()),
    "rows with NO usable text (subject+body)": int((df.text == "").sum()),
    "rows sharing subject+body with another row": int(df.duplicated(["subject", "body"], keep=False).sum()),
    "bodies at the 999-char truncation limit": int((raw["body"].str.len() >= 990).sum()),
    "bodies mentioning 1999-2002": int(df.body.str.contains(r"\\b(?:1999|2000|2001|2002)\\b").sum()),
    "bodies mentioning 2010 or 2011": int(df.body.str.contains(r"\\b(?:2010|2011)\\b").sum()),
}, name="count")
quality
""")
code("""
msgs_per_month = df.groupby("month").size()
print("Messages per month: min", msgs_per_month.min(), "| max", msgs_per_month.max())
print("Employee-months with at least one message:", df.groupby(["employee", "month"]).ngroups, "of", df.employee.nunique() * df.month.nunique())
df.groupby("employee").size().sort_values(ascending=False).rename("messages").to_frame()
""")
md("""
**Observations (data quality - important).**
* No *formal* missing values, but **243 messages have no subject**, **32 have no usable text** at all
  (these are scored Neutral), and two bodies hit the 999-character export limit, i.e. bodies are
  **truncated** (sentiment late in long mails is invisible).
* **The `date` column looks synthetic.** Every month contains exactly 91-92 messages (an almost
  perfectly even spread), and **576 bodies mention 1999-2002 while none mention 2010 or 2011** - the
  original mails are from ~2000-2001 and the dates appear to have been reassigned. All "trends over
  time" below are therefore trends **over the dates in the file**, not necessarily real chronology.
* **About 49% of rows (1,074) share their subject and body with at least one other row** - mostly the
  same mail appearing under several senders (only 82 rows are same-sender repeats). We keep them
  (each is a message that employee sent) and test the effect on flight risk in section 5.
* One sender (`bobette.riner@ipgdirect.com`) is an external domain; the other nine are `@enron.com`.
  All 10 senders appear in all 24 months, so no employee-month is missing.
""")

md("### 2.2 Distribution of sentiment")
code("""
fig, ax = plt.subplots(figsize=(7, 2.8))
vals = df["sentiment"].value_counts().reindex(ORDER[::-1])
bars = ax.barh(vals.index, vals.values, color=[C[k] for k in vals.index], height=0.6)
for b, v in zip(bars, vals.values):
    ax.text(v + 15, b.get_y() + b.get_height() / 2, f"{v:,}  ({v / len(df):.1%})", va="center", color=C["ink"])
ax.set_xlim(0, vals.max() * 1.25); ax.grid(axis="y", visible=False)
ax.set_title("Most messages are neutral; negatives are under 10%"); ax.set_xlabel("Number of messages")
save(fig, "01_sentiment_distribution.png")
""")
md("""
**Observations.** Neutral 1,382 (63.1%), Positive 646 (29.5%), Negative 163 (7.4%). Because
negatives are rare, *four or more in a month window* (the flight-risk rule) is a high bar.
""")

md("### 2.3 Trends over time")
code("""
mon = df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)[ORDER]
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
bottom = np.zeros(len(mon))
for lab in ORDER:
    axes[0].bar(mon.index, mon[lab], bottom=bottom, color=C[lab], label=lab, edgecolor=C["surface"], linewidth=1.2)
    bottom += mon[lab].values
axes[0].set_title("Message volume is flat (91-92 per month) - the mix, not the volume, changes")
axes[0].set_ylabel("Messages"); axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02)) if False else axes[0].legend(ncol=3, loc="lower right")
net = df["sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1}).groupby(df["month"]).mean()
x = np.arange(len(net)); slope, icpt = np.polyfit(x, net.values, 1)
axes[1].plot(net.index, net.values, color=C["Positive"], lw=2, marker="o", ms=4, label="Mean score per message")
axes[1].plot(net.index, icpt + slope * x, color=C["ink2"], lw=1.2, ls="--", label=f"Linear trend ({slope:+.4f}/month)")
axes[1].set_title("Average sentiment per message drifts upward"); axes[1].set_ylabel("Mean score (+1 / 0 / -1)")
axes[1].legend(loc="upper left")
plt.setp(axes[1].get_xticklabels(), rotation=60, ha="right"); fig.tight_layout()
save(fig, "02_monthly_volume_and_trend.png")
from scipy import stats
print(stats.linregress(x, net.values))
""")
md("""
**Observations.** Volume is essentially constant (see data-quality note), so changes come from the
*mix*. The mean score per message is ~0.14-0.19 in the first months of 2010 and ~0.2-0.33 in the last
months of 2011; the fitted slope is about **+0.0046 per month** (r = 0.63, p = 0.001) - a statistically
visible but modest drift towards more positive wording. Monthly values still bounce around (0.14 to 0.33), so no single month
should be over-interpreted.
""")

md("### 2.4 Sentiment by employee")
code("""
share = pd.crosstab(df["employee"], df["sentiment"], normalize="index")[ORDER] * 100
share = share.sort_values("Positive")
fig, ax = plt.subplots(figsize=(10, 5))
left = np.zeros(len(share))
for lab in ORDER:
    ax.barh([short(e) for e in share.index], share[lab], left=left, color=C[lab], label=lab, edgecolor=C["surface"], linewidth=1.5)
    for y, (l, v) in enumerate(zip(left, share[lab])):
        if v > 6: ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white" if lab != "Neutral" else C["ink"], fontsize=8)
    left += share[lab].values
ax.set_xlim(0, 100); ax.grid(axis="y", visible=False); ax.set_xlabel("% of the employee's messages")
ax.set_title("Eric Bass is the most positive writer, Rhonda Denton the most neutral"); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
save(fig, "03_sentiment_by_employee.png")
share.round(1)
""")
md("""
**Observations.** Differences between people are modest: positive share ranges from ~22% (Rhonda
Denton) to ~35% (Eric Bass) and negative share from ~6% (John Arnold) to ~9% (Kayne Coulter).
Rhonda Denton and Patti Thompson have the lowest positive shares (22% and 26%) and also write mostly
neutral mail (70% and 65%).
""")

md("### 2.5 Other patterns: message length, and employee x month intensity")
code("""
df["word_count"] = df["body_clean"].str.split().str.len().fillna(0)
fig, ax = plt.subplots(figsize=(8, 4))
data = [df.loc[df.sentiment == l, "word_count"].clip(upper=250) for l in ORDER]
bp = ax.boxplot(data, vert=False, tick_labels=ORDER, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color=C["ink"], lw=1.5))
for patch, l in zip(bp["boxes"], ORDER): patch.set_facecolor(C[l]); patch.set_edgecolor(C["surface"])
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("Words in cleaned body (capped at 250)")
ax.set_title("Positive and negative mails are about twice as long as neutral ones")
save(fig, "04_length_by_sentiment.png")
df.groupby("sentiment")["word_count"].describe().round(1).loc[ORDER]
""")
md("""
**Observations.** Median length is ~21 words for Neutral versus ~44-48 for Positive/Negative. Part of
this is mechanical: a bag-of-words scorer has more chances to hit a sentiment word in a longer text.
Short logistics mails ("see attached", "game at 7") are almost always Neutral.
""")

md("## 3. Employee score calculation (Task 3)")
md("""
**Method.** Each message gets a score: **Positive = +1, Negative = -1, Neutral = 0**. Scores are
summed per *employee x calendar month* (`groupby(["month", "employee"])`). The score "resets" every
month automatically, because a message only ever belongs to the month in which it was sent - there is
no running total carried over. An employee who sent nothing in a month has no row; in this dataset
that never happens (all 240 employee-months exist).
""")
code("""
df = add_message_score(df)
scores = monthly_scores(df)
print(scores.shape, "->", scores.employee.nunique(), "employees x", scores.month.nunique(), "months")

# --- validation 1: totals must reconcile with the message-level scores
assert scores["monthly_score"].sum() == df["message_score"].sum()
assert scores["n_messages"].sum() == len(df)
assert (scores["n_positive"] - scores["n_negative"] == scores["monthly_score"]).all()
# --- validation 2: re-compute one employee-month by hand
chk = df[(df.employee == "sally.beck@enron.com") & (df.month == "2010-05")]
print("Sally Beck 2010-05 by hand:", chk["message_score"].sum(), "| table:",
      scores.query("employee=='sally.beck@enron.com' and month=='2010-05'")["monthly_score"].iloc[0])
scores.head(10)
""")
code("""
pivot = scores.pivot(index="employee", columns="month", values="monthly_score")
pivot.index = [short(e) for e in pivot.index]
cmap = LinearSegmentedColormap.from_list("div", [C["Negative"], "#ecebe6", C["Positive"]])
lim = np.abs(pivot.values).max()
fig, ax = plt.subplots(figsize=(14, 4.8))
im = ax.imshow(pivot.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index); ax.grid(False)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=7.5, color=C["ink"])
cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8); cb.set_label("Monthly score (positive - negative)")
ax.set_title("Monthly sentiment score per employee (blue = net positive, orange = net negative)")
save(fig, "05_employee_month_heatmap.png")
print("Employee-months with a negative score:", int((scores.monthly_score < 0).sum()), "of", len(scores))
print("Score range:", scores.monthly_score.min(), "to", scores.monthly_score.max())
""")
md("""
**Observations.** Scores are small integers because most messages are neutral. Net-negative
employee-months are rare (see count above) and most cells are mildly positive, mirroring the overall
mix. The maximum monthly score is driven as much by *how many* messages someone sent as by tone - this
matters for the regression in section 6.
""")

md("## 4. Employee ranking (Task 4)")
md("""
**Method.** Using the monthly scores from section 3, for each month:
* **Top-3 positive** - the three highest scores, sorted by score *descending*, ties broken
  *alphabetically* by employee.
* **Top-3 negative** - the three *lowest* scores, most negative first, ties broken alphabetically.

Because every month has ten employees and scores are small integers, ties are common; the
alphabetical rule makes the output deterministic, but it means a tie at the cut-off is decided by
name, not by evidence. We also give a whole-period ranking (sum of the 24 monthly scores) for the
headline "top 3".
""")
code("""
top_pos, top_neg = rank_monthly(scores, top_n=3)
for d in (top_pos, top_neg): d["name"] = d["employee"].map(short)
print("Example - 2010-01 and 2011-12")
display_cols = ["month", "rank", "name", "monthly_score"]
pd.concat([top_pos[top_pos.month.isin(["2010-01", "2011-12"])][display_cols].assign(list="Top positive"),
           top_neg[top_neg.month.isin(["2010-01", "2011-12"])][display_cols].assign(list="Top negative")])
""")
code("""
overall = overall_ranking(scores)
overall["score_per_message"] = (overall["total_score"] / overall["n_messages"]).round(3)
overall["name"] = overall["employee"].map(short)
overall[["name", "employee", "total_score", "n_messages", "score_per_message", "n_positive", "n_negative"]]
""")
code("""
fig, ax = plt.subplots(figsize=(9, 5))
o = overall.sort_values("total_score")
colors = [C["Positive"] if i >= len(o) - 3 else (C["Negative"] if i < 3 else C["Neutral"]) for i in range(len(o))]
bars = ax.barh(o["name"], o["total_score"], color=colors, height=0.65)
for b, v in zip(bars, o["total_score"]): ax.text(v + 1, b.get_y() + b.get_height() / 2, str(v), va="center")
ax.grid(axis="y", visible=False); ax.set_xlabel("Total score over 24 months (sum of monthly scores)")
ax.set_title("Overall ranking: top 3 (blue) and bottom 3 (orange) by total score")
save(fig, "06_overall_ranking.png")
""")
code("""
# How often does each employee appear in a monthly top-3 / bottom-3?
app = pd.DataFrame({"months in top-3 positive": top_pos.groupby("employee").size(),
                    "months in top-3 negative": top_neg.groupby("employee").size()}).fillna(0).astype(int)
app.index = [short(e) for e in app.index]; app = app.sort_values("months in top-3 positive")
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(app.index, app["months in top-3 positive"], color=C["Positive"], height=0.65, label="Top-3 positive")
ax.barh(app.index, -app["months in top-3 negative"], color=C["Negative"], height=0.65, label="Top-3 negative (lowest scores)")
ax.axvline(0, color=C["ink2"], lw=0.8); ax.grid(axis="y", visible=False)
ax.set_xlabel("Number of months (out of 24)"); ax.legend(loc="lower right")
ax.set_title("Who shows up in the monthly top-3 lists")
save(fig, "07_monthly_top3_appearances.png")
app.sort_values("months in top-3 positive", ascending=False)
""")
md("""
**Observations.**
* **Top three positive (whole period):** Lydia Delgado (66), Eric Bass (60), John Arnold (59).
* **Three lowest:** Rhonda Denton (25), Kayne Coulter (37), Patti Thompson (38).
* **Careful reading:** *all ten totals are positive*, so "lowest" means *least positive*, not
  "unhappy". Totals also scale with message count (Lydia sent the most, Rhonda the fewest). Per message,
  Eric Bass is the most positive (0.286) and Rhonda Denton (0.145) and Patti Thompson (0.169) are the least -
  the per-message column is the fairer measure of tone; the total is the measure the brief asks for.
* Monthly lists change a lot from month to month (tiny integer scores, many ties), so a monthly
  top-3 is a noisy signal; the whole-period view and the appearance counts are more stable.
""")

md("## 5. Flight-risk identification (Task 5)")
md("""
**Definition (from the brief).** A *flight risk* is any employee who sent **4 or more negative
e-mails within 30 days**, irrespective of score and irrespective of month boundaries.

**Method.** For each employee we take their Negative messages and slide a **rolling 30-day window**
(`pandas rolling("30D")`, i.e. the day itself plus the previous 29 days). Whenever the count reaches 4
the employee is flagged and the trigger date is recorded. Choices worth knowing:
* Messages on the same day all count (4 negatives on one day is enough).
* "30 days" means 30 calendar days inclusive: negatives on 1 Jan and 30 Jan are in one window, 1 Jan
  and 31 Jan are not.
* **Robustness:** the result is cross-checked against an independent brute-force implementation, and
  boundary / month-crossing cases are unit-tested in `tests/test_pipeline.py`.
""")
code("""
triggers = flight_risk_windows(df)
risk = flight_risk_summary(df)
risk["name"] = risk["employee"].map(short)

# robustness 1: independent brute-force implementation must flag exactly the same people
assert set(risk.employee) == flight_risk_bruteforce(df), "implementations disagree!"
print("Flagged (rolling-window method) == flagged (brute force):", sorted(risk.employee))
risk[["name", "employee", "first_flagged", "last_flagged", "max_negatives_in_30d", "trigger_days", "months_affected"]]
""")
code("""
# show the actual negative e-mails inside each employee's FIRST trigger window (evidence for the flag)
neg = df[df.sentiment == "Negative"]
for _, r in risk.iterrows():
    first = triggers[triggers.employee == r.employee].sort_values("trigger_date").iloc[0]
    w = neg[(neg.employee == r.employee) & (neg.date >= first.window_start) & (neg.date <= first.trigger_date)]
    print(f"\\n{r['name']}: {len(w)} negative mails between {first.window_start.date()} and {first.trigger_date.date()}")
    for _, m in w.iterrows(): print(f"   {m.date.date()}  [{m.sentiment_score:+.2f}]  {m.text[:95]}")
""")
code("""
# robustness 2: do duplicated mails (same sender, same subject+body) drive the flags?
dedup = df.drop_duplicates(["employee", "subject", "body"])
risk_dedup = flight_risk_summary(dedup)
print(f"Rows: {len(df)} -> {len(dedup)} after dropping same-sender duplicates")
print("Flagged without duplicates:", sorted(risk_dedup.employee))
print("Flagged only WITH duplicates:", sorted(set(risk.employee) - set(risk_dedup.employee)))

# robustness 3: sensitivity to the negative threshold
for t in (0.15, 0.25, 0.35):
    tmp = df.copy(); tmp["sentiment"] = tmp["sentiment_score"].map(lambda s: label_from_score(s, t, -t))
    print(f"threshold +-{t}: flagged = {sorted(short(e) for e in flight_risk_summary(tmp).employee)}")
""")
code("""
# visual: rolling 30-day negative count per employee, threshold = 4
days = pd.date_range(df.date.min(), df.date.max())
emps = sorted(df.employee.unique())
fig, axes = plt.subplots(2, 5, figsize=(16, 5.6), sharex=True, sharey=True)
for ax, e in zip(axes.ravel(), emps):
    d = df[(df.employee == e) & (df.sentiment == "Negative")].groupby("date").size().reindex(days, fill_value=0)
    roll = d.rolling("30D").sum()
    flagged = e in set(risk.employee)
    ax.plot(roll.index, roll.values, color=C["Negative"] if flagged else C["Neutral"], lw=1.6)
    ax.axhline(4, color=C["ink2"], ls="--", lw=1)
    ax.fill_between(roll.index, 4, roll.values, where=roll.values >= 4, color=C["Negative"], alpha=0.35)
    ax.set_title(short(e) + ("  - FLAGGED" if flagged else ""), fontsize=9.5, color=C["ink"])
    ax.tick_params(axis="x", rotation=45, labelsize=7.5)
axes[0, 0].set_ylim(0, 5.5); fig.suptitle("Negative e-mails in the trailing 30 days (dashed line = flight-risk threshold of 4)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); save(fig, "08_flight_risk_timeline.png")
""")
md("""
**Observations.**
* **One employee is flagged: Bobette Riner** - 4 negative e-mails between 30 Dec 2010 and 28 Jan 2011
  (the rule triggers again on 17 Feb 2011 as the window slides). She reached **exactly 4**, never more, so the
  flag is *marginal*: one fewer negative label and she would not be flagged.
* The four messages behind the flag are mixed: a "Transition Issues" note and an "I regret to inform you
  ... call has been canceled" notice are mildly negative business mails; one is fantasy-football banter
  ("your running backs suck"); one is a long deal e-mail with a few cautionary words. Read them before
  drawing any conclusion about the person.
* The rolling-window method and the brute-force check agree, and the window evidence list shows
  the actual messages behind each flag so a human can judge them.
* **The list is fragile.** Removing same-sender duplicates changes nothing, but the label threshold
  matters a lot: at +-0.15 **seven** employees are flagged, at +-0.25 (our setting) **one**, at +-0.35
  **none**. An earlier version of the lexicon (before the boilerplate and "risk"/"outage" fixes described in
  Section 1) flagged four people. So this is a **"worth a conversation"** signal, not proof that someone
  is about to leave - the underlying "negative" label is lexicon-based.
* A short negative burst is independent of the total score, which is why the brief defines the flag
  separately from the ranking.
""")

md("## 6. Predictive modelling - linear regression (Task 6)")
md("""
**Goal.** Explore whether simple, *non-sentiment* characteristics of how people write predict their
monthly sentiment score.

**Data.** One row per employee-month (240 rows). **Features** (deliberately not the sentiment words
themselves, to avoid leaking the label): `n_messages` (message frequency), `avg_word_count`,
`avg_char_length`, `avg_subject_words`, `share_questions` (share of mails containing "?"),
`share_exclaims` (share containing "!"), `month_index` (0..23, a time trend).

**Two targets.**
* **A. `monthly_score`** - the Task-3 score. Because it is a *sum*, it mechanically grows with the
  number of messages, so we expect `n_messages` to dominate.
* **B. `mean_sentiment`** - score divided by message count, i.e. tone *independent of volume*.

**Evaluation.** 80/20 random train/test split (seed 42), metrics on the held-out 20%
(R-squared, RMSE, MAE) compared with a "predict the training mean" baseline, plus 5-fold
cross-validation on the training set. Coefficient p-values come from the usual OLS t-tests (training data).
""")
code("""
table = build_model_table(df, scores)
FEATURES = ["n_messages", "avg_word_count", "avg_char_length", "avg_subject_words",
            "share_questions", "share_exclaims", "month_index"]
print(table.shape); table[FEATURES + ["monthly_score", "mean_sentiment"]].describe().round(2).T
""")
code("""
resA = fit_linear_model(table, FEATURES, "monthly_score")
resB = fit_linear_model(table, FEATURES, "mean_sentiment")
metrics = pd.DataFrame({"A: monthly_score": resA["metrics"], "B: mean_sentiment": resB["metrics"]}).round(3)
metrics
""")
code("""
print("Model A - monthly_score (coefficients on TRAINING data)")
display_A = resA["coefs"].round(4); display_A
""")
code("""
print("Model B - mean_sentiment (coefficients on TRAINING data)")
resB["coefs"].round(4)
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, res, ttl in ((axes[0], resA, "A: monthly score"), (axes[1], resB, "B: mean sentiment per message")):
    y, p = res["y_test"], res["pred_test"]
    ax.scatter(p, y, s=28, color=C["Positive"], alpha=0.75, edgecolor=C["surface"], linewidth=0.8)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color=C["ink2"], ls="--", lw=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (test set)")
    ax.set_title(f"{ttl}  (test R2 = {res['metrics']['r2_test']:.2f})", fontsize=10.5)
resid = resA["y_test"] - resA["pred_test"]
axes[2].hist(resid, bins=14, color=C["Neutral"], edgecolor=C["surface"]); axes[2].axvline(0, color=C["ink2"], ls="--", lw=1)
axes[2].set_title("Model A test residuals", fontsize=10.5); axes[2].set_xlabel("Actual - predicted")
fig.tight_layout(); save(fig, "09_model_actual_vs_predicted.png")
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
for ax, res, ttl in ((axes[0], resA, "A: monthly score"), (axes[1], resB, "B: mean sentiment")):
    c = res["coefs"].iloc[1:].set_index("feature")["std_coef"].sort_values()
    sig = res["coefs"].iloc[1:].set_index("feature")["p_value"].reindex(c.index) < 0.05
    ax.barh(c.index, c.values, color=[C["Positive"] if s else C["Neutral"] for s in sig], height=0.62)
    ax.axvline(0, color=C["ink2"], lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_title(f"{ttl}: standardised coefficients", fontsize=10.5); ax.set_xlabel("Effect of +1 s.d. in feature (in s.d. of target)")
fig.suptitle("Blue = statistically significant at 5%, grey = not", x=0.01, ha="left", color=C["ink2"], fontsize=9.5)
fig.tight_layout(); save(fig, "10_model_coefficients.png")
""")
md("""
**Observations - what the model does and does not tell us.**
* **Model A (monthly score)** explains a moderate share of variance on unseen data
  (test R-squared about 0.48, test RMSE about 1.48 vs 2.06 for the mean-only baseline; 5-fold CV
  R-squared is lower at ~0.28, so the test figure is partly luck of the split). **`n_messages` is by far the
  strongest coefficient** (~+0.17 score points per extra message, p < 0.001). That is largely
  *arithmetic*: more messages -> a bigger sum. It is a real but unsurprising relationship, not evidence that
  busy people are happier. `month_index` is also significant at the 5% level (p = 0.028, ~+0.035 points per month), echoing the
  slow upward drift in section 2.3.
* **Model B (tone per message)** has **essentially no predictive power**: test R-squared is about 0.05 and CV R-squared is
  negative (-0.11) - it does no better than guessing the average. Two features reach p < 0.05:
  `avg_subject_words` (longer subject lines -> slightly lower sentiment) and `month_index` (the same slow
  upward drift), but with seven features tested and no out-of-sample gain these are weak evidence.
* **Interpretation.** Simple structural features (length, punctuation, frequency) do **not**
  forecast how positive someone's mail will be once volume is controlled for. A better model would
  need content features (topics, recipients, seniority, deal calendar) or genuine time-ordered data.
* **Caveats.** 240 observations, a lexicon-generated target (label noise flows straight into
  the regression), the synthetic-looking dates, and a linear model on bounded, discrete scores.
  Residuals (right-hand panel above) are centred near zero but right-skewed: a handful of very high-score months are under-predicted. The fit is adequate for a baseline, not for forecasting.
""")

md("## 7. Summary & exports")
code("""
# ---- export all result tables ----
df_out = df.drop(columns=["word_count"], errors="ignore")
df_out[["employee", "date", "month", "subject", "body", "sentiment_score", "sentiment", "message_score"]] \\
    .to_csv(os.path.join(ROOT, "data", "test_labeled.csv"), index=False)
scores.to_csv(os.path.join(OUT, "monthly_scores.csv"), index=False)
top_pos.to_csv(os.path.join(OUT, "monthly_top3_positive.csv"), index=False)
top_neg.to_csv(os.path.join(OUT, "monthly_top3_negative.csv"), index=False)
overall.to_csv(os.path.join(OUT, "overall_ranking.csv"), index=False)
risk.to_csv(os.path.join(OUT, "flight_risks.csv"), index=False)
triggers.to_csv(os.path.join(OUT, "flight_risk_trigger_days.csv"), index=False)
resA["coefs"].assign(model="A: monthly_score").pipe(lambda a: pd.concat([a, resB["coefs"].assign(model="B: mean_sentiment")])) \\
    .to_csv(os.path.join(OUT, "model_coefficients.csv"), index=False)
metrics.to_csv(os.path.join(OUT, "model_metrics.csv"))
print("Saved:", sorted(os.listdir(OUT)), "+ data/test_labeled.csv")

# ---- headline results ----
print("\\nTOP 3 POSITIVE (total score):", ", ".join(f"{short(e)} ({s})" for e, s in overall.head(3)[["employee", "total_score"]].values))
print("TOP 3 LOWEST   (total score):", ", ".join(f"{short(e)} ({s})" for e, s in overall.tail(3).iloc[::-1][["employee", "total_score"]].values))
print("FLIGHT RISKS:", ", ".join(risk["name"]))
""")
md("""
### Key insights & recommendations
1. **Overall tone is healthy and drifting up** - 30% positive vs 7% negative, with a small upward trend.
   Negative e-mail is rare, so concentrated bursts of negativity stand out.
2. **One employee is flagged** (Bobette Riner): exactly four negatives in 30 days. This is a *marginal*
   flag from a lexicon labeller, and the list changes with the threshold (7 people at +-0.15, none at
   +-0.35) - use it to start a conversation, not to conclude anything. Read the evidence messages first.
3. **Rankings show small differences** - all ten employees have positive totals. Rhonda Denton, Kayne Coulter
   and Patti Thompson score lowest; for Rhonda and Kayne this is partly lower message volume, while Patti sent
   a typical number of messages and has a lower per-message score.
4. **Do not use the regression to forecast sentiment.** Volume explains the *score*; nothing structural
   explains the *tone*.
5. **Before anyone acts on this:** validate labels against human-labelled data (or an LLM) and
   verify the date column - the synthetic-looking dates limit any time-based conclusion.
""")

# ============================================================================ EXECUTOR
def _png_of_open_figures():
    import matplotlib.pyplot as plt
    outs = []
    for num in plt.get_fignums():
        buf = io.BytesIO(); plt.figure(num).savefig(buf, format="png", dpi=100, bbox_inches="tight")
        outs.append(base64.b64encode(buf.getvalue()).decode()); plt.close(num)
    return outs

def _stream(text): return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}

def execute_cells():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {"__name__": "__main__"}
    os.chdir(os.path.join(ROOT, "notebooks"))
    # make plt.show() a no-op placeholder: figures are collected after each cell instead
    nb_cells, count = [], 0
    for kind, src in CELLS:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}); continue
        count += 1
        outputs, buf = [], io.StringIO()
        tree = ast.parse(src); last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        value = None
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, "<cell>", "exec"), ns)
                plt.show = lambda *a, **k: None
                if last is not None: value = eval(compile(last, "<cell>", "eval"), ns)
        except Exception:
            print(f"\n!!! cell {count} failed:\n{src}\n"); traceback.print_exc(); raise
        if buf.getvalue(): outputs.append(_stream(buf.getvalue()))
        for png in _png_of_open_figures():
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": png, "text/plain": ["<Figure>"]}})
        if value is not None:
            import pandas as pd
            data = {"text/plain": str(value).splitlines(keepends=True)}
            if hasattr(value, "to_html"):
                data["text/html"] = value.to_html(max_rows=40, max_cols=20).splitlines(keepends=True)
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
                         "source": src.splitlines(keepends=True)})
    return nb_cells

if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    cells = execute_cells()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as fh: json.dump(nb, fh, indent=1)
    print("Notebook written to", NB_PATH)
