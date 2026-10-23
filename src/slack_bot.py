"""Slack bot that answers company policy questions using the RAG approach.

Runs locally over Socket Mode (no public URL needed). From the project folder:
    python -m src.slack_bot

Needs in .env:
    SLACK_BOT_TOKEN=xoxb-...   (OAuth & Permissions -> Bot User OAuth Token)
    SLACK_APP_TOKEN=xapp-...   (Basic Information -> App-Level Tokens, scope connections:write)
    GEMINI_API_KEY=...
"""
import logging
import os
import re

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from src.llm_rag import LLMRAGSearcher
from src.policies import DECLINED, SUPPORTED

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = App(token=os.environ["SLACK_BOT_TOKEN"])
searcher = LLMRAGSearcher()


def answer_question(question: str) -> str:
    question = question.strip()
    if not question:
        return "Ask me a company policy question, e.g. `@Policy Assistant How many vacation days do I get?`"

    res = searcher.search(question)
    if res["grounding"] == SUPPORTED:
        return (
            f"*Answer:* {res['answer']}\n"
            f"*Relevant policy:* {res['policy']} ({res['category']})"
        )
    if res["grounding"] == DECLINED:
        return "I couldn't find a company policy that covers this question."
    if res["grounding"] == "Error":
        return "Sorry, the language model is unavailable right now. Please try again in a moment."
    return f"*Answer:* {res['answer']}\n_⚠️ This answer could not be linked to a company policy._"


@app.event("app_mention")
def handle_mention(event, say):
    """Someone wrote @Policy Assistant <question> in a channel."""
    question = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", ""))
    say(text=answer_question(question), thread_ts=event.get("thread_ts", event["ts"]))


@app.event("message")
def handle_direct_message(event, say):
    """Direct messages to the bot. Channel messages without a mention are ignored."""
    if event.get("channel_type") != "im" or event.get("bot_id") or event.get("subtype"):
        return
    say(text=answer_question(event.get("text", "")))


@app.command("/policy")
def handle_slash_command(ack, command, respond):
    """/policy <question>, answered visibly in the channel."""
    ack()
    question = command.get("text", "")
    respond(
        response_type="in_channel",
        text=f"*Question:* {question}\n{answer_question(question)}",
    )


if __name__ == "__main__":
    print("Policy bot running. Press Ctrl+C to stop.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
