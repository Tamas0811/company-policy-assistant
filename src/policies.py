"""Shared helpers: loading the policy database and checking whether an answer is grounded in it."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "company_policies.csv"

NO_MATCH = "No matching company policy found."

# Grounding labels shown on the website and used in the evaluation
SUPPORTED = "Supported"        # answer cites a policy that exists in the database
DECLINED = "Declined"          # approach said no policy covers the question
UNSUPPORTED = "Unsupported"    # approach answered without a valid policy behind it


def load_policies(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    for col in ["title", "department", "policy_text", "category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def policy_lookup(df: pd.DataFrame) -> dict:
    """Map lower-cased title -> row (as dict) for validating policies an LLM cites."""
    return {row["title"].lower(): row for row in df.to_dict("records")}


def parse_llm_output(text: str):
    """Parse the 'POLICY: ... / ANSWER: ...' format both LLM approaches are asked to use."""
    text = (text or "").replace("**", "").strip()  # models sometimes add markdown bold
    policy_match = re.search(r"POLICY:\s*(.+)", text, flags=re.IGNORECASE)
    answer_match = re.search(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    policy = policy_match.group(1).strip().strip('"*') if policy_match else ""
    answer = answer_match.group(1).strip() if answer_match else text
    return policy, answer


def is_decline(answer: str) -> bool:
    return "no matching company policy" in (answer or "").lower()


def check_grounding(cited_policy: str, answer: str, lookup: dict):
    """Return (grounding label, matched policy row or None)."""
    # tolerate "Vacation Policy (Human Resources)" or a trailing full stop
    title = re.sub(r"\s*\(.*\)\s*$", "", (cited_policy or "")).strip().rstrip(".").lower()
    row = lookup.get(title)
    if is_decline(answer):
        return DECLINED, None
    if row is not None:
        return SUPPORTED, row
    return UNSUPPORTED, None
