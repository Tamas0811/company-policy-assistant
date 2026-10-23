# Company Policy Assistant

Answers company policy questions and names the relevant policy, comparing three approaches:

| Approach | File | How it works |
|---|---|---|
| Rules-based search | `src/rules_search.py` | BM25 keyword ranking; says "no policy found" below a score threshold |
| LLM without vector index | `src/llm_direct.py` | Gemini receives **all** policies in the prompt |
| LLM with vector index (RAG) | `src/llm_rag.py` | Chroma retrieves the 3 closest policies; only those go to Gemini |

The Streamlit site (`app.py`) runs all three side by side and shows answer, policy, response time,
token use and a grounding check. `src/evaluate.py` measures accuracy and unsupported answers on a fixed
question set. `src/slack_bot.py` is the Slack bot (uses RAG).

## 1. Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # then put your GEMINI_API_KEY in .env
streamlit run app.py
```

If Gemini returns "model not found", run the notebook cell that lists models and set
`GEMINI_MODELS=<name1>,<name2>` in `.env`.

## 2. Run the evaluation (before deploying)

```bash
python -m src.evaluate
```

This takes a couple of minutes (it waits between calls to stay under the free-tier rate limit) and writes
`evaluation_results.csv`, which the website shows as a table. Commit that file. Then check that the two
paragraphs at the bottom of `app.py` still match your numbers.

## 3. Deploy (Streamlit Community Cloud)

1. Push this folder to a **public** GitHub repo (`.gitignore` keeps `.env`, `venv/` and `chroma_db/` out).
2. Go to share.streamlit.io → *Create app* → pick the repo, branch `main`, main file `app.py`.
3. *Advanced settings → Secrets*:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy. The first load builds the vector index (downloads a small embedding model), so it takes a minute.

## 4. Slack bot

1. Create a **new Slack workspace** named so it identifies you (e.g. *Tamas Kiszely – Policy Bot*), and a channel such as
   `#policy-bot-tamas-kiszely`.
2. Invite **raz@sdu.dk** to the workspace and that channel.
3. api.slack.com/apps → *Create New App* → *From a manifest* → choose your workspace → paste `slack_manifest.yml`.
4. *Basic Information → App-Level Tokens* → generate a token with scope `connections:write` → this is `SLACK_APP_TOKEN` (`xapp-…`).
5. *Install App* → install to workspace → copy the *Bot User OAuth Token* → `SLACK_BOT_TOKEN` (`xoxb-…`).
6. Put both tokens in `.env`, then run:
   ```bash
   python -m src.slack_bot
   ```
7. In the channel type `/invite @Policy Assistant`, then ask:
   `@Policy Assistant How many vacation days do I get?` (or use `/policy How many vacation days do I get?`).
8. Take the screenshot showing the question and the bot's answer with the policy name.
