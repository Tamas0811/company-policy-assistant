"""Benchmark the three approaches on a fixed question set.

Run from the project folder:   python -m src.evaluate
It writes evaluation_results.csv (summary, shown on the website) and
evaluation_details.csv (every question x approach).
"""
import argparse
import time

import pandas as pd

from src.llm_direct import LLMDirectSearcher
from src.llm_rag import LLMRAGSearcher
from src.policies import DECLINED, ROOT
from src.rules_search import RulesSearcher

# Questions the policy database DOES answer (with the expected policy).
# Worded the way an employee would ask, not copied from the policy text.
IN_SCOPE = [
    ("How many vacation days do I get each year?", "Vacation Policy"),
    ("How often do I need to change my password?", "Password Policy"),
    ("How long can I stay home after having a baby?", "Parental Leave Policy"),
    ("How many days a week can I work from home?", "Remote Work Policy"),
    ("When do I have to hand in receipts after a business trip?", "Travel Reimbursement Policy"),
    ("What should I wear to the office?", "Dress Code Policy"),
    ("Can I use my work laptop for personal things?", "Laptop Usage Policy"),
    ("How long is the trial period for new hires?", "Probation Period Policy"),
    ("Can I report unethical behaviour without giving my name?", "Whistleblower Policy"),
    ("Can I get an ergonomic chair for my home office?", "Home Office Equipment Policy"),
]

# Questions the policy database does NOT answer. Any answer other than
# "no matching policy" counts as an answer unsupported by the database.
OUT_OF_SCOPE = [
    "Can I bring my dog to the office?",
    "What is the company's current share price?",
    "How big is the annual salary increase?",
    "Is lunch in the canteen free?",
    "What is the maximum hotel price per night on business trips?",
    "How many weeks of notice do I need to give before resigning?",
]


def run(delay: float):
    engines = {
        "Rules-based (BM25)": RulesSearcher(),
        "LLM without index": LLMDirectSearcher(),
        "LLM with vector index (RAG)": LLMRAGSearcher(),
    }
    questions = [(q, exp, True) for q, exp in IN_SCOPE] + [(q, None, False) for q in OUT_OF_SCOPE]

    rows = []
    for q, expected, in_scope in questions:
        for name, engine in engines.items():
            res = engine.search(q)
            rows.append({
                "approach": name,
                "question": q,
                "in_scope": in_scope,
                "expected_policy": expected or "",
                "returned_policy": res["policy"],
                "grounding": res["grounding"],
                "answer": res["answer"],
                "response_time_s": res["response_time"],
                "total_tokens": res["token_usage"]["total_tokens"],
            })
            print(f"[{name}] {q} -> {res['policy']} ({res['grounding']})")
            if name != "Rules-based (BM25)":
                time.sleep(delay)  # stay under the free-tier rate limit

    details = pd.DataFrame(rows)
    details["correct_policy"] = details["in_scope"] & (
        details["returned_policy"].str.lower() == details["expected_policy"].str.lower()
    )
    # Unsupported answer = answered an out-of-scope question, or gave an answer with no valid policy behind it
    details["unsupported_answer"] = ((~details["in_scope"]) & (details["grounding"] != DECLINED)) | (
        details["grounding"] == "Unsupported"
    )

    summary = []
    for name in engines:
        d = details[details["approach"] == name]
        ins, outs = d[d["in_scope"]], d[~d["in_scope"]]
        summary.append({
            "Approach": name,
            "Correct policy (in-scope questions)": f"{ins['correct_policy'].sum()}/{len(ins)}",
            "Unsupported answers (out-of-scope questions)": f"{outs['unsupported_answer'].sum()}/{len(outs)}",
            "Avg. response time (s)": round(d["response_time_s"].mean(), 3),
            "Avg. tokens per question": int(round(d["total_tokens"].mean())),
            "Errors": int((d["grounding"] == "Error").sum()),
        })
    summary = pd.DataFrame(summary)

    details.to_csv(ROOT / "evaluation_details.csv", index=False)
    summary.to_csv(ROOT / "evaluation_results.csv", index=False)
    print("\n", summary.to_string(index=False))
    return summary, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=4.0, help="seconds to wait between LLM calls")
    run(parser.parse_args().delay)
