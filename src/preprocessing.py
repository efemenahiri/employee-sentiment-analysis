"""Loading and cleaning of the raw employee e-mail dataset (data/test.csv)."""
import re
import pandas as pd

# Matches the header block that starts a quoted / forwarded earlier message.
_FORWARD_MARKERS = re.compile(
    r"(-{2,}\s*original message\s*-{2,}|-{2,}\s*forwarded by|"
    r"^\s*from:\s.+\n\s*(sent|to|date):)",
    re.IGNORECASE | re.MULTILINE,
)
_URL = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_QP_SOFT_BREAK = re.compile(r"=\r?\n")           # quoted-printable line continuation
_QP_HEX = re.compile(r"=[0-9A-Fa-f]{2}")           # quoted-printable escapes: =20, =01 ...
# Boilerplate that is not the sender's own words: legal/confidentiality footers and
# closing salutations ("Best regards" would otherwise count as the positive word "best").
_DISCLAIMER = re.compile(r"the information transmitted is intended only.*", re.IGNORECASE | re.DOTALL)
_SIGNOFF = re.compile(r"\b(best|kind|warm|warmest|many)\s+regards\b|\bregards\b", re.IGNORECASE)
_SUBJECT_PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.IGNORECASE)


def load_raw(path: str) -> pd.DataFrame:
    """Read test.csv. The file is tab-separated (despite the .csv name) and has
    quoted multi-line message bodies; we auto-detect tab vs comma from the header."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    df = pd.read_csv(path, sep=sep, quotechar='"', encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_body(text: str) -> str:
    """Remove artefacts so only text actually written by the sender is scored."""
    if not isinstance(text, str):
        return ""
    text = _QP_SOFT_BREAK.sub("", text)
    text = text.replace("=20", " ")
    text = _QP_HEX.sub(" ", text)
    m = _FORWARD_MARKERS.search(text)           # drop quoted/forwarded history
    if m:
        text = text[: m.start()]
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(">"))
    text = _DISCLAIMER.sub("", text)
    text = _SIGNOFF.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_subject(text: str) -> str:
    if not isinstance(text, str) or text.strip().lower() in {"(no subject)", "no subject"}:
        return ""
    return _SUBJECT_PREFIX.sub("", text).strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates (month/day/year), add cleaned text and month columns."""
    out = df.copy()
    out = out.rename(columns={"Subject": "subject", "from": "employee"})
    out["date"] = pd.to_datetime(out["date"], format="%m/%d/%Y", errors="coerce")
    out["subject_clean"] = out["subject"].map(clean_subject)
    out["body_clean"] = out["body"].map(clean_body)
    out["text"] = (out["subject_clean"] + ". " + out["body_clean"]).str.strip(" .")
    out["month"] = out["date"].dt.to_period("M").astype(str)
    return out
