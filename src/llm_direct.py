"""Approach 2: LLM without a vector index.

Every question is sent together with the *entire* policy database, so there is no
retrieval step. This shows what an index saves: the model must find the right policy
itself, and every question pays for all policies in tokens.
"""
import time

from src.gemini_client import EMPTY_USAGE, GeminiClient
from src.policies import NO_MATCH, UNSUPPORTED, check_grounding, load_policies, parse_llm_output, policy_lookup

PROMPT = """You are an official company policy assistant.
Answer the employee's question using ONLY the company policies listed below.
If none of the policies answers the question, the ANSWER must be exactly: "{no_match}" and POLICY must be NONE.

Reply in exactly this format:
POLICY: <exact title of the policy you used, or NONE>
ANSWER: <short answer>

Company policies:
{policies}

Question: {query}
"""


class LLMDirectSearcher:
    def __init__(self):
        self.llm = GeminiClient()
        self.df = load_policies()
        self.lookup = policy_lookup(self.df)
        self.policy_block = "\n".join(
            f"- {r['title']} ({r['category']}): {r['policy_text']}" for r in self.df.to_dict("records")
        )

    def search(self, query: str):
        start = time.time()
        prompt = PROMPT.format(no_match=NO_MATCH, policies=self.policy_block, query=query)
        try:
            text, token_usage, model = self.llm.generate(prompt)
        except RuntimeError as e:
            return {
                "answer": f"The model could not be reached, please try again. ({e})",
                "policy": "Error", "category": "N/A",
                "response_time": round(time.time() - start, 2),
                "token_usage": dict(EMPTY_USAGE), "grounding": "Error", "model": None,
            }

        cited, answer = parse_llm_output(text)
        grounding, row = check_grounding(cited, answer, self.lookup)
        return {
            "answer": answer,
            "policy": row["title"] if row else (cited if grounding == UNSUPPORTED and cited else "None"),
            "category": row["category"] if row else "N/A",
            "response_time": round(time.time() - start, 2),
            "token_usage": token_usage,
            "grounding": grounding,
            "model": model,
        }
