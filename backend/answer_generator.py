import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

PERSONA_INTENTS = {
    "what kind of person": "persona_overview",
    "who is this user": "persona_overview",
    "describe the user": "persona_overview",
    "type of person": "persona_overview",
    "what are their habits": "habits",
    "what habits": "habits",
    "how do they talk": "comm_style",
    "communication style": "comm_style",
    "how do they communicate": "comm_style",
    "what do they talk about": "topics",
    "interests": "topics",
    "personality": "personality",
    "traits": "personality",
    "personal facts": "personal_facts",
    "occupation": "personal_facts",
    "job": "personal_facts",
}


class AnswerGenerator:

    def answer(self, query: str, retrieved: Dict, persona: Dict) -> str:
        intent = self._detect_intent(query)
        persona_summary = self._build_persona_text(persona)
        context = retrieved.get("context_text", "")
        raw_msgs = retrieved.get("raw_messages", [])
        msgs_text = "\n".join(
            f"{m['sender']}: {m['text']}" for m in raw_msgs[:10]
        )

        prompt = f"""You are an AI analyst that has studied a user's conversations.

USER PERSONA (extracted from all conversations):
{persona_summary}

RETRIEVED CONVERSATION SEGMENTS (most relevant to the query):
{context}

SAMPLE RAW MESSAGES:
{msgs_text}

USER QUESTION: {query}

Answer clearly and specifically based on the evidence above.
Use bullet points where helpful. Be concise but insightful."""

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_answer(intent, persona, query)

    def _detect_intent(self, query: str) -> Optional[str]:
        q = query.lower()
        for phrase, intent in PERSONA_INTENTS.items():
            if phrase in q:
                return intent
        return None

    def _build_persona_text(self, p: Dict) -> str:
        habits = p.get("habits", {}).get("detected", [])
        traits = p.get("personality_traits", {}).get("dominant_traits", [])
        topics = p.get("top_topics", [])
        facts = p.get("personal_facts", {})
        cs = p.get("communication_style", {})
        lines = []
        if facts:
            for k, v in facts.items():
                lines.append(f"- {k}: {', '.join(str(x) for x in v[:3])}")
        if traits:
            lines.append(f"- Personality traits: {', '.join(traits)}")
        if habits:
            lines.append(f"- Habits: {', '.join(h.replace('_',' ') for h in habits[:8])}")
        if topics:
            lines.append(f"- Frequent topics: {', '.join(topics[:5])}")
        lines.append(f"- Communication tone: {cs.get('tone', 'neutral')}")
        lines.append(f"- Message style: {cs.get('message_style', 'medium')}")
        return "\n".join(lines)

    def _fallback_answer(self, intent, persona, query):
        return f"Based on the conversations, here is what I found for: {query}\n\n{self._build_persona_text(persona)}"