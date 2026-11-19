"""Scoring, ranking, flight-risk detection and regression for the employee
sentiment project.  Every function is pure (DataFrame in -> DataFrame out) so it
can be unit-tested (see tests/test_pipeline.py) and re-used from the notebook."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

SCORE_MAP = {"Positive": 1, "Negative": -1, "Neutral": 0}
FLIGHT_RISK_MIN_NEGATIVES = 4
FLIGHT_RISK_WINDOW_DAYS = 30


# --------------------------------------------------------------------------- Task 3
def add_message_score(df: pd.DataFrame) -> pd.DataFrame:
    """Positive = +1, Negative = -1, Neutral = 0."""
    out = df.copy()
    out["message_score"] = out["sentiment"].map(SCORE_MAP).astype(int)
    return out


def monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of message scores per employee per calendar month.

    The score "resets" simply because we group by (employee, calendar month):
    a message only ever contributes to the month in which it was sent.
    Employees who sent nothing in a month have no row (no evidence = no score)."""
    g = df.groupby(["month", "employee"])
    out = g.agg(
        monthly_score=("message_score", "sum"),
        n_messages=("message_score", "size"),
        n_positive=("sentiment", lambda s: (s == "Positive").sum()),
        n_negative=("sentiment", lambda s: (s == "Negative").sum()),
        n_neutral=("sentiment", lambda s: (s == "Neutral").sum()),
    ).reset_index()
    return out.sort_values(["month", "employee"]).reset_index(drop=True)


# --------------------------------------------------------------------------- Task 4
def rank_monthly(scores: pd.DataFrame, top_n: int = 3):
    """Per month: top-N most positive and top-N most negative employees.

    Positive list: score descending, ties broken alphabetically.
    Negative list: score ascending (most negative first), ties alphabetical.
    Returns (top_positive, top_negative) long-format DataFrames with a `rank` column."""
    pos_rows, neg_rows = [], []
    for month, grp in scores.groupby("month"):
        p = grp.sort_values(["monthly_score", "employee"], ascending=[False, True]).head(top_n)
        n = grp.sort_values(["monthly_score", "employee"], ascending=[True, True]).head(top_n)
        for r, (_, row) in enumerate(p.iterrows(), 1):
            pos_rows.append({"month": month, "rank": r, "employee": row.employee,
                             "monthly_score": row.monthly_score})
        for r, (_, row) in enumerate(n.iterrows(), 1):
            neg_rows.append({"month": month, "rank": r, "employee": row.employee,
                             "monthly_score": row.monthly_score})
    return pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)


def overall_ranking(scores: pd.DataFrame) -> pd.DataFrame:
    """Whole-period ranking: total of the monthly scores per employee."""
    t = scores.groupby("employee").agg(
        total_score=("monthly_score", "sum"),
        months_active=("month", "nunique"),
        n_messages=("n_messages", "sum"),
        n_positive=("n_positive", "sum"),
        n_negative=("n_negative", "sum"),
    ).reset_index()
    t["mean_monthly_score"] = (t.total_score / t.months_active).round(2)
    return t.sort_values(["total_score", "employee"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- Task 5
def flight_risk_windows(df: pd.DataFrame, min_neg: int = FLIGHT_RISK_MIN_NEGATIVES,
                        window_days: int = FLIGHT_RISK_WINDOW_DAYS) -> pd.DataFrame:
    """Rolling-window flight-risk detection.

    An employee is a flight risk if, for ANY window of `window_days` consecutive
    calendar days (rolling, not tied to month boundaries), they sent >= `min_neg`
    negative e-mails.  A window of 30 days starting on day D covers D .. D+29, so
    k negatives are in one window iff first and last are <= 29 days apart.

    Returns one row per employee x day on which the rolling count reached the
    threshold (the "trigger" days), with the rolling count and window start."""
    neg = df[df["sentiment"] == "Negative"].copy()
    rows = []
    for emp, grp in neg.groupby("employee"):
        # one row per negative message, indexed by date; rolling window (t-30d, t]
        s = pd.Series(1, index=grp["date"].sort_values().values)
        # collapse same-day messages, then rolling sum over a 30-day time window
        daily = s.groupby(level=0).sum()
        rolled = daily.rolling(f"{window_days}D").sum()
        for day, cnt in rolled.items():
            if cnt >= min_neg:
                rows.append({"employee": emp, "trigger_date": day, "negatives_in_window": int(cnt),
                             "window_start": day - pd.Timedelta(days=window_days - 1)})
    return pd.DataFrame(rows, columns=["employee", "trigger_date", "negatives_in_window", "window_start"])


def flight_risk_summary(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """One row per flagged employee: first trigger date, worst 30-day count, #trigger days."""
    w = flight_risk_windows(df, **kw)
    if w.empty:
        return pd.DataFrame(columns=["employee", "first_flagged", "last_flagged",
                                     "max_negatives_in_30d", "trigger_days", "months_affected"])
    g = w.groupby("employee").agg(
        first_flagged=("trigger_date", "min"),
        last_flagged=("trigger_date", "max"),
        max_negatives_in_30d=("negatives_in_window", "max"),
        trigger_days=("trigger_date", "nunique"),
        months_affected=("trigger_date", lambda s: ", ".join(sorted(s.dt.strftime("%Y-%m").unique()))),
    ).reset_index()
    return g.sort_values(["max_negatives_in_30d", "employee"], ascending=[False, True]).reset_index(drop=True)


def flight_risk_bruteforce(df: pd.DataFrame, min_neg: int = 4, window_days: int = 30) -> set:
    """Independent slow implementation used to cross-check `flight_risk_windows`."""
    flagged = set()
    neg = df[df["sentiment"] == "Negative"]
    for emp, grp in neg.groupby("employee"):
        dates = grp["date"].dt.normalize()
        for day in pd.date_range(dates.min(), dates.max()):
            n = ((dates >= day - pd.Timedelta(days=window_days - 1)) & (dates <= day)).sum()
            if n >= min_neg:
                flagged.add(emp)
                break
    return flagged


# --------------------------------------------------------------------------- Task 6
def build_model_table(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Employee-month feature table.  Features describe HOW and HOW MUCH people write
    (never the sentiment words themselves, which would leak the label)."""
    d = df.copy()
    d["body_words"] = d["body_clean"].str.split().str.len().fillna(0)
    d["body_chars"] = d["body_clean"].str.len()
    d["subject_words"] = d["subject_clean"].str.split().str.len().fillna(0)
    d["has_question"] = d["body_clean"].str.contains(r"\?", regex=True).astype(int)
    d["has_exclaim"] = d["body_clean"].str.contains("!", regex=False).astype(int)
    feats = d.groupby(["month", "employee"]).agg(
        avg_word_count=("body_words", "mean"),
        avg_char_length=("body_chars", "mean"),
        avg_subject_words=("subject_words", "mean"),
        share_questions=("has_question", "mean"),
        share_exclaims=("has_exclaim", "mean"),
    ).reset_index()
    t = scores.merge(feats, on=["month", "employee"])
    months = sorted(t["month"].unique())
    t["month_index"] = t["month"].map({m: i for i, m in enumerate(months)})
    t["mean_sentiment"] = t["monthly_score"] / t["n_messages"]   # score per message
    return t


def ols_summary(X: np.ndarray, y: np.ndarray, names, model: LinearRegression) -> pd.DataFrame:
    """Coefficient table with standard errors, t-stats and p-values (fit on training data)."""
    n, k = X.shape
    pred = model.predict(X)
    resid = y - pred
    dof = n - k - 1
    sigma2 = (resid @ resid) / dof
    Xd = np.column_stack([np.ones(n), X])
    cov = sigma2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    beta = np.concatenate([[model.intercept_], model.coef_])
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    out = pd.DataFrame({"feature": ["intercept"] + list(names), "coef": beta, "std_err": se,
                        "t": t, "p_value": p})
    # standardised coefficient = coef * std(x) / std(y): comparable effect sizes
    out["std_coef"] = [np.nan] + list(model.coef_ * X.std(axis=0, ddof=1) / y.std(ddof=1))
    return out


def fit_linear_model(table: pd.DataFrame, features, target: str, test_size=0.2, seed=42):
    """Train/test split + LinearRegression + metrics. Returns a dict of everything needed."""
    X = table[features].to_numpy(dtype=float)
    y = table[target].to_numpy(dtype=float)
    idx = np.arange(len(table))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    model = LinearRegression().fit(X[tr], y[tr])
    pred_tr, pred_te = model.predict(X[tr]), model.predict(X[te])
    cv = cross_val_score(LinearRegression(), X[tr], y[tr], cv=5, scoring="r2")
    metrics = {
        "n_train": len(tr), "n_test": len(te),
        "r2_train": r2_score(y[tr], pred_tr), "r2_test": r2_score(y[te], pred_te),
        "rmse_test": float(np.sqrt(mean_squared_error(y[te], pred_te))),
        "mae_test": mean_absolute_error(y[te], pred_te),
        "baseline_rmse_test": float(np.sqrt(mean_squared_error(y[te], np.full(len(te), y[tr].mean())))),
        "cv_r2_mean": cv.mean(), "cv_r2_std": cv.std(),
    }
    return {"model": model, "metrics": metrics, "train_idx": tr, "test_idx": te,
            "y_test": y[te], "pred_test": pred_te, "y_train": y[tr], "pred_train": pred_tr,
            "coefs": ols_summary(X[tr], y[tr], features, model)}
