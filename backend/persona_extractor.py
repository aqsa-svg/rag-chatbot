"""
persona_extractor.py
─────────────────────
Extracts structured user persona from conversation data.
ALL signals grounded in actual text patterns — no guessing.

Output JSON schema:
{
  "habits": { ... },
  "personal_facts": { ... },
  "personality_traits": { ... },
  "communication_style": { ... },
  "top_topics": [ ... ],
  "evidence_count": int
}
"""

import re
import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any


# ──────────────────────────────────────────────────────────────
# PATTERN BANKS  (regex → trait)
# ──────────────────────────────────────────────────────────────

HABIT_PATTERNS: Dict[str, List[str]] = {
    "coffee_drinker":       [r"\bcoffee\b", r"\bespresso\b", r"\blatte\b", r"\bcaffeine\b"],
    "tea_drinker":          [r"\btea\b", r"\bchai\b", r"\bgreen tea\b"],
    "gamer":                [r"\bgam(e|ing|ed)\b", r"\bplayer\b", r"\blevel up\b", r"\bquest\b", r"\bxbox\b", r"\bplaystation\b", r"\bsteam\b"],
    "reader":               [r"\bbook(s)?\b", r"\bnovel\b", r"\breading\b", r"\blibrarian\b", r"\bkindle\b", r"\bauthor\b"],
    "fitness_enthusiast":   [r"\bgym\b", r"\bworkout\b", r"\bexercise\b", r"\brunning\b", r"\bweights\b", r"\bcardio\b", r"\byoga\b", r"\bprotein\b"],
    "foodie":               [r"\brestaurant\b", r"\brecipe\b", r"\bcook(ing)?\b", r"\bchef\b", r"\beat(ing)?\b", r"\bcuisine\b", r"\bfood\b"],
    "movie_watcher":        [r"\bmovie\b", r"\bfilm\b", r"\bnetflix\b", r"\bbinge\b", r"\bcinema\b", r"\bseries\b", r"\bwatch\b"],
    "music_lover":          [r"\bmusic\b", r"\bsong\b", r"\bplaylist\b", r"\bband\b", r"\bconcert\b", r"\balbum\b", r"\bspotify\b"],
    "traveler":             [r"\btravel\b", r"\btrip\b", r"\bvacation\b", r"\bflight\b", r"\bcountry\b", r"\bbackpack\b", r"\bhotel\b"],
    "pet_owner":            [r"\bdog\b", r"\bcat\b", r"\bpet\b", r"\bpuppy\b", r"\bkitten\b"],
    "night_owl":            [r"\bstay(ing)? up\b", r"\blate night\b", r"\bmidnight\b", r"\bcan't sleep\b", r"\binsomnia\b", r"\bup all night\b"],
    "early_riser":          [r"\bwoke up early\b", r"\bgood morning\b", r"\bup at \d?[456]am\b", r"\bearly bird\b"],
    "outdoors_person":      [r"\bhike\b", r"\bhiking\b", r"\bcamping\b", r"\bnature\b", r"\btrail\b", r"\boutdoors\b"],
    "social_media_user":    [r"\btwitter\b", r"\binstagram\b", r"\btiktok\b", r"\bpost\b", r"\bstory\b", r"\bfollower\b"],
    "driver":               [r"\bcar\b", r"\bdrive\b", r"\bdriving\b", r"\bimpala\b", r"\bvehicle\b"],
    "student":              [r"\bstudent\b", r"\bschool\b", r"\bcollege\b", r"\buniversity\b", r"\bclass\b", r"\bexam\b", r"\bhomework\b", r"\bdegree\b"],
}

PERSONALITY_PATTERNS: Dict[str, List[str]] = {
    "humorous":     [r"\bhaha\b", r"\blol\b", r"\blmao\b", r"\bfunny\b", r"\bjoke\b", r"\bjk\b", r"\bcringe\b", r"\brofl\b"],
    "emotional":    [r"\bfeel(ing)?\b", r"\bhurt\b", r"\bcry(ing)?\b", r"\bsad\b", r"\bexcited\b", r"\banxious\b", r"\boverwhelmed\b"],
    "sarcastic":    [r"\boh great\b", r"\bsure sure\b", r"\byeah right\b", r"\bobviously\b", r"\bclearly\b"],
    "analytical":   [r"\bbecause\b", r"\btherefore\b", r"\blogically\b", r"\btechnically\b", r"\bactually\b", r"\bprecisely\b", r"\banalysis\b"],
    "empathetic":   [r"\bi understand\b", r"\bi feel you\b", r"\bi hear you\b", r"\bthat sucks\b", r"\bi'm sorry\b", r"\bsending love\b"],
    "introverted":  [r"\balone\b", r"\bby myself\b", r"\bquiet\b", r"\bintrovert\b", r"\bsocial anxiety\b", r"\bstay home\b"],
    "extroverted":  [r"\bparty\b", r"\bpeople\b", r"\bhangout\b", r"\boutgoing\b", r"\bsocial\b", r"\bfriends\b"],
    "anxious":      [r"\bworried\b", r"\banxiety\b", r"\bpanic\b", r"\bnervous\b", r"\boverthinking\b", r"\bwhat if\b", r"\bstressed\b"],
    "optimistic":   [r"\bhopeful\b", r"\bpositive\b", r"\bbright side\b", r"\bbelieve\b", r"\bgood vibes\b", r"\bit'll be okay\b"],
    "adventurous":  [r"\badventure\b", r"\bexplore\b", r"\bthrill\b", r"\bdaring\b", r"\brisk\b", r"\bnew experience\b"],
    "creative":     [r"\bcreative\b", r"\bart\b", r"\bpaint\b", r"\bwrite\b", r"\bdesign\b", r"\bcreate\b", r"\bimagine\b"],
    "caring":       [r"\bcare\b", r"\blove\b", r"\bhelp(ing)?\b", r"\bsupport\b", r"\bwarm\b", r"\bkind\b"],
}

COMMUNICATION_PATTERNS: Dict[str, str] = {
    "uses_emojis":        r"[\U0001F300-\U0001FFFF\u2600-\u26FF\u2700-\u27BF]",
    "uses_caps_emphasis": r"\b[A-Z]{3,}\b",
    "uses_ellipsis":      r"\.\.\.",
    "uses_profanity":     r"\b(wtf|damn|hell|shit|fuck|crap|bitch)\b",
    "uses_abbreviations": r"\b(tbh|imo|idk|omg|ngl|lmk|btw|fwiw|smh|brb|afk|irl|fyi)\b",
    "asks_questions":     r"\?",
    "uses_exclamations":  r"!",
    "uses_lol":           r"\b(lol|lmao|haha|hehe)\b",
}

PERSONAL_FACT_PATTERNS = {
    "occupation": [
        (r"\bi(?:'m| am) a[n]? (\w+ \w+|\w+)\b", 1),
        (r"\bwork as a[n]? (\w+)\b", 1),
        (r"\bmy job is (\w+)\b", 1),
        (r"\bi(?:'m| am) a (\w+ \w+ \w+|\w+ \w+|\w+) by profession\b", 1),
    ],
    "location": [
        (r"\bi(?:'m| am)? (?:from|in|living in|based in|moving to) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", 1),
    ],
    "relationship_status": [
        (r"\b(single|married|divorced|engaged|in a relationship|dating)\b", 0),
        (r"\bmy (wife|husband|partner|boyfriend|girlfriend|spouse)\b", 1),
    ],
    "family": [
        (r"\bmy (kids?|children|son|daughter|siblings?|brother|sister|mom|dad|mother|father|parents?|family)\b", 1),
    ],
    "hobbies_explicit": [
        (r"\bi love (?:to )?([\w ]+)", 1),
        (r"\bi enjoy (?:to )?([\w ]+)", 1),
        (r"\bmy hobbies? (?:is|are|include) ([\w ,]+)", 1),
        (r"\bi(?:'m| am) into ([\w ]+)", 1),
    ],
    "age_mentions": [
        (r"\bi(?:'m| am) (\d{1,2}) years old\b", 1),
        (r"\bage (\d{1,2})\b", 1),
    ],
}


# ──────────────────────────────────────────────────────────────
# EXTRACTOR
# ──────────────────────────────────────────────────────────────

class PersonaExtractor:

    def __init__(self):
        pass

    def extract(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Extract persona from a flat list of message dicts.
        Each dict needs at least 'text' and optionally 'sender'.
        """
        all_texts = [m["text"] for m in messages]
        combined = " ".join(all_texts).lower()
        total = len(messages)

        habits = self._extract_habits(combined, all_texts)
        personality = self._extract_personality(combined, all_texts, total)
        comm_style = self._extract_comm_style(messages)
        personal_facts = self._extract_personal_facts(all_texts)
        top_topics = self._extract_top_topics(combined)
        avg_msg_length = sum(len(t.split()) for t in all_texts) / max(total, 1)

        comm_style["avg_message_length_words"] = round(avg_msg_length, 1)
        comm_style["message_style"] = (
            "very short" if avg_msg_length < 6 else
            "short" if avg_msg_length < 12 else
            "medium" if avg_msg_length < 20 else
            "detailed / long-form"
        )

        return {
            "summary": self._build_summary(habits, personality, comm_style, personal_facts),
            "habits": habits,
            "personal_facts": personal_facts,
            "personality_traits": personality,
            "communication_style": comm_style,
            "top_topics": top_topics,
            "evidence_count": total
        }

    # ── Habits ────────────────────────────────────────────────

    def _extract_habits(self, combined: str, texts: List[str]) -> Dict:
        counts = {}
        for habit, patterns in HABIT_PATTERNS.items():
            c = sum(
                len(re.findall(p, combined, re.IGNORECASE))
                for p in patterns
            )
            if c >= 2:  # require at least 2 mentions to be confident
                counts[habit] = c

        # Sort by frequency
        sorted_habits = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        return {
            "detected": list(sorted_habits.keys()),
            "mention_counts": sorted_habits
        }

    # ── Personality ───────────────────────────────────────────

    def _extract_personality(self, combined: str, texts: List[str], total: int) -> Dict:
        scores = {}
        for trait, patterns in PERSONALITY_PATTERNS.items():
            c = sum(
                len(re.findall(p, combined, re.IGNORECASE))
                for p in patterns
            )
            # Normalize as mentions per 100 messages
            if c >= 2:
                scores[trait] = round((c / total) * 100, 2)

        sorted_traits = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        dominant = list(sorted_traits.keys())[:5]
        return {
            "dominant_traits": dominant,
            "trait_scores_per_100_msgs": sorted_traits
        }

    # ── Communication Style ───────────────────────────────────

    def _extract_comm_style(self, messages: List[Dict]) -> Dict:
        total = len(messages)
        counts = defaultdict(int)

        for m in messages:
            text = m["text"]
            for feat, pattern in COMMUNICATION_PATTERNS.items():
                if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
                    counts[feat] += 1

        rates = {
            feat: round((cnt / total) * 100, 1)
            for feat, cnt in counts.items()
        }

        # Tone inference
        tone = "neutral"
        if rates.get("uses_exclamations", 0) > 40 and rates.get("uses_emojis", 0) > 20:
            tone = "enthusiastic & expressive"
        elif rates.get("uses_lol", 0) > 15:
            tone = "casual & playful"
        elif rates.get("asks_questions", 0) > 50:
            tone = "inquisitive & engaged"
        elif rates.get("uses_caps_emphasis", 0) > 10:
            tone = "emphatic / passionate"
        elif rates.get("uses_ellipsis", 0) > 15:
            tone = "thoughtful / trailing"

        return {
            "tone": tone,
            "feature_rates_pct": rates,
            "emoji_user": rates.get("uses_emojis", 0) > 10,
            "abbreviation_user": rates.get("uses_abbreviations", 0) > 5,
            "question_heavy": rates.get("asks_questions", 0) > 40,
        }

    # ── Personal Facts ────────────────────────────────────────

    def _extract_personal_facts(self, texts: List[str]) -> Dict:
        facts: Dict[str, list] = defaultdict(list)

        for text in texts:
            for fact_type, pat_list in PERSONAL_FACT_PATTERNS.items():
                for (pattern, group_idx) in pat_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for m in matches:
                        val = m if isinstance(m, str) else m[group_idx] if group_idx < len(m) else m[0]
                        val = val.strip().lower()
                        # Filter noise
                        if len(val) > 2 and val not in ("the", "a", "an", "my", "me"):
                            if val not in facts[fact_type]:
                                facts[fact_type].append(val)

        # Limit to top 5 per category
        return {k: v[:5] for k, v in facts.items() if v}

    # ── Top Topics ────────────────────────────────────────────

    def _extract_top_topics(self, combined: str) -> List[str]:
        topic_keywords = {
            "work & career": ["job","work","career","office","boss","meeting","colleague","salary","promotion","business"],
            "relationships & family": ["family","parents","mom","dad","kids","boyfriend","girlfriend","husband","wife","partner","friends","relationship"],
            "health & fitness": ["gym","workout","exercise","health","doctor","sick","diet","yoga","run","fitness","weight"],
            "food & cooking": ["food","eat","cook","recipe","restaurant","dinner","lunch","breakfast","meal","cuisine"],
            "entertainment": ["movie","music","game","netflix","show","concert","book","read","watch","play"],
            "travel & places": ["travel","trip","vacation","city","country","flight","hotel","visit","move","place"],
            "education": ["school","college","university","study","class","degree","exam","learn","teacher","student"],
            "technology": ["phone","computer","app","tech","software","internet","social media","online","code","program"],
            "hobbies & interests": ["hobby","interest","passion","enjoy","love","fun","sport","art","creative","practice"],
            "emotions & mental health": ["feel","emotion","anxiety","stress","happy","sad","worried","overwhelmed","peace","therapy"],
        }

        topic_scores = {}
        for topic, kws in topic_keywords.items():
            score = sum(combined.count(w) for w in kws)
            if score > 0:
                topic_scores[topic] = score

        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_topics[:6]]

    # ── Summary ───────────────────────────────────────────────

    def _build_summary(self, habits, personality, comm_style, facts) -> str:
        parts = []

        # Occupation
        occ = facts.get("occupation", [])
        loc = facts.get("location", [])
        if occ:
            parts.append(f"Works as a {occ[0]}")
        if loc:
            parts.append(f"based in or connected to {loc[0]}")

        # Dominant personality
        traits = personality.get("dominant_traits", [])
        if traits:
            parts.append(f"personality leans {', '.join(traits[:3])}")

        # Top habits
        detected_habits = habits.get("detected", [])
        if detected_habits:
            parts.append(f"notable habits: {', '.join(detected_habits[:4])}")

        # Comm style
        tone = comm_style.get("tone", "neutral")
        parts.append(f"communication tone is {tone}")

        return ". ".join(parts).capitalize() + "." if parts else "Insufficient data for summary."


# ──────────────────────────────────────────────────────────────
# SAVE / LOAD
# ──────────────────────────────────────────────────────────────

def save_persona(persona: Dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(f"[Persona] Saved to {out_path}")


def load_persona(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
