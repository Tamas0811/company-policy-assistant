"""Approach 3: LLM with a vector index (RAG).

The question is embedded, the closest policies are retrieved from a Chroma vector
index, and only those are sent to the LLM.
"""
import time

import chromadb

from src.gemini_client import EMPTY_USAGE, GeminiClient
from src.policies import (NO_MATCH, ROOT, UNSUPPORTED, check_grounding, load_policies, parse_llm_output,
                          policy_lookup)

TOP_K = 3  # number of policies retrieved per question

PROMPT = """You are an official company policy assistant.
Answer the employee's question using ONLY the policy context below.
If the context does not answer the question, the ANSWER must be exactly: "{no_match}" and POLICY must be NONE.

Reply in exactly this format:
POLICY: <exact title of the policy you used, or NONE>
ANSWER: <short answer>

Policy context:
{context}

Question: {query}
"""


class LLMRAGSearcher:
    def __init__(self):
        self.llm = GeminiClient()
        self.df = load_policies()
        self.lookup = policy_lookup(self.df)

        client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
        self.collection = client.get_or_create_collection(name="policies")
        # (Re)build the index if it is empty or out of date with the CSV
        if self.collection.count() != len(self.df):
            self._build_vector_index()

    def _build_vector_index(self):
        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
        records = self.df.to_dict("records")
        self.collection.add(
            ids=[f"policy_{i}" for i in range(len(records))],
            documents=[
                f"Title: {r['title']}\nCategory: {r['category']}\nDepartment: {r['department']}\nContent: {r['policy_text']}"
                for r in records
            ],
            metadatas=[{k: r[k] for k in ["title", "category", "department", "policy_text"]} for r in records],
        )

    def retrieve(self, query: str, k: int = TOP_K):
        res = self.collection.query(query_texts=[query], n_results=k)
        return res["documents"][0], res["metadatas"][0]

    def search(self, query: str):
        start = time.time()
        docs, metas = self.retrieve(query)
        retrieved = [m["title"] for m in metas]
        context = "\n\n".join(docs)
        prompt = PROMPT.format(no_match=NO_MATCH, context=context, query=query)

        try:
            text, token_usage, model = self.llm.generate(prompt)
        except RuntimeError as e:
            return {
                "answer": f"The model could not be reached, please try again. ({e})",
                "policy": retrieved[0] if retrieved else "Error",
                "category": metas[0]["category"] if metas else "N/A",
                "response_time": round(time.time() - start, 2),
                "token_usage": dict(EMPTY_USAGE), "grounding": "Error",
                "retrieved": retrieved, "model": None,
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
            "retrieved": retrieved,
            "model": model,
        }
