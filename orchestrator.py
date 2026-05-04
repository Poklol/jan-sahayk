from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
import google.generativeai as genai

from memory import get_missing_required_fields, next_followup_question

FRESHNESS_KEYWORDS = {"latest", "today", "new", "recent", "update", "updated", "current"}
DETAIL_KEYWORDS = {
    "detail",
    "detailed",
    "explain",
    "explanation",
    "eligible",
    "eligibility",
    "can i apply",
    "which scheme",
    "which schemes",
    "best scheme",
}

STATE_KEYWORDS = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "jammu", "ladakh", "puducherry",
]

OCCUPATION_HINTS = {
    "farmer": ["farmer", "agriculture", "kisan"],
    "student": ["student", "studying", "college", "school"],
    "laborer": ["labor", "labour", "worker", "daily wage"],
    "self-employed": ["self employed", "business", "shop", "vendor", "entrepreneur"],
    "unemployed": ["unemployed", "jobless", "no job"],
}

CATEGORY_HINTS = {
    "SC": ["sc", "scheduled caste"],
    "ST": ["st", "scheduled tribe"],
    "OBC": ["obc"],
    "EWS": ["ews"],
    "General": ["general"],
}

OCCUPATION_ALIASES = {
    "farming": "farmer",
    "farm": "farmer",
    "farmer": "farmer",
    "famring": "farmer",
    "agriculture": "farmer",
    "agri": "farmer",
}

GREETING_HINTS = {
    "hi",
    "hello",
    "hey",
    "vanakkam",
    "namaste",
    "good morning",
    "good evening",
    "good afternoon",
}

STATE_ALIASES = {
    "kerela": "kerala",
    "orissa": "odisha",
    "pondicherry": "puducherry",
}

SYSTEM_PROMPT = """
You are an AI Welfare Advisor for Indian citizens.

Your goals:
1. Sound natural, warm, and conversational (NOT robotic)
2. Keep responses SHORT and SCANNABLE
3. Always explain WHY a scheme is recommended
4. Always guide the user on WHAT TO DO NEXT
5. Never dump paragraphs

RESPONSE STYLE RULES:

- Start with a friendly acknowledgment (e.g., "Got it 👍")
- Use simple language (rural-friendly English)
- Use bullet points instead of paragraphs
- Keep each section max 3 bullets
- Add emojis for clarity (₹, ✔, 📋, 🔗)

STRUCTURE:

🎯 Recommended for You

[Scheme Name]

✔ Eligibility Status (based on user profile)

💰 Benefits:
• ...
• ...

📋 Why this matches you:
• ...
• ...

🪜 What to do next:
• Step 1
• Step 2

🔗 Apply here:
[official link]
"""

STATES = {
    "RECOMMENDING": "recommending",
    "SHOWING_DOCS": "showing_docs",
    "APPLICATION_HELP": "application_help",
}

NEED_KEYWORDS = {
    "loan_support": ["loan", "credit", "finance", "financial assistance", "borrow"],
    "scholarship_support": ["scholarship", "education", "student", "study"],
    "pension_support": ["pension", "old age", "retirement", "senior"],
    "housing_support": ["housing", "house", "home", "shelter", "awas"],
}


class JanSahayakOrchestrator:
    def __init__(self, rag_agent, web_agent) -> None:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
        self.rag_agent = rag_agent
        self.web_agent = web_agent
        self.model = None

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            configure_fn = getattr(genai, "configure", None)
            model_cls = getattr(genai, "GenerativeModel", None)
            if callable(configure_fn) and model_cls is not None:
                configure_fn(api_key=api_key)
                self.model = model_cls("gemini-1.5-flash")

        self.top_k_rag = int(os.getenv("TOP_K_RAG", "4"))
        self.top_k_web = int(os.getenv("TOP_K_WEB", "3"))
        self.last_scheme: Dict[str, Any] | None = None
        self.last_ranked_schemes: List[Dict[str, Any]] = []
        self.session: Dict[str, Any] = {
            "last_intent": None,
            "last_scheme": None,
            "last_schemes": [],
            "current_step": None,
            "primary_need": "",
        }

    def _is_greeting_text(self, text: str) -> bool:
        normalized = re.sub(r"[^a-zA-Z\s]", " ", (text or "").lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return False
        if normalized in GREETING_HINTS:
            return True
        # Handles short variants like "hi sir", "hello ai", "vanakkam anna"
        tokens = normalized.split()
        if tokens and tokens[0] in {"hi", "hello", "hey", "vanakkam", "namaste"}:
            return True
        return False

    def _normalize_state(self, raw_state: str) -> str:
        cleaned = re.sub(r"\s+", " ", (raw_state or "").strip().lower())
        if not cleaned:
            return ""
        if cleaned in STATE_ALIASES:
            cleaned = STATE_ALIASES[cleaned]
        return cleaned

    def _pretty_state(self, normalized_state: str) -> str:
        if not normalized_state:
            return ""
        return normalized_state.title()

    def _normalize_occupation(self, raw_occupation: str) -> str:
        cleaned = re.sub(r"\s+", " ", (raw_occupation or "").strip().lower())
        if not cleaned:
            return ""
        if cleaned in OCCUPATION_ALIASES:
            return OCCUPATION_ALIASES[cleaned]
        for alias, canonical in OCCUPATION_ALIASES.items():
            if alias in cleaned:
                return canonical
        return cleaned

    def detect_intent(self, message: Any, session: Dict[str, Any]) -> str:
        if isinstance(message, dict):
            raw_intent = str(message.get("intent", "")).strip().upper()
            mapping = {
                "SHOW_DOCUMENTS": "documents",
                "APPLICATION_HELP": "application_help",
                "MORE_SCHEMES": "more_schemes",
                "NEXT_STEP": "next_step",
                "STOP": "completed",
                "NEW_QUERY": "new_query",
            }
            return mapping.get(raw_intent, "scheme_recommendation")

        msg = (message or "").lower().strip()

        if msg in ["yes", "yeah", "ok", "okay", "sure"]:
            if session.get("current_step") == STATES["SHOWING_DOCS"]:
                return "application_help"
            if session.get("current_step") == STATES["APPLICATION_HELP"]:
                return "completed"

        if any(x in msg for x in ["document", "documents", "papers", "required", "docs"]):
            return "documents"
        if any(x in msg for x in ["eligibility", "eligible", "requirements", "criteria"]):
            return "eligibility_help"
        if any(x in msg for x in ["apply", "application", "steps", "how to apply", "process"]):
            return "application_help"
        if any(x in msg for x in ["new", "restart", "another question"]):
            return "new_query"
        if any(x in msg for x in ["more", "other", "another", "show more"]):
            return "more_schemes"

        if "help" in msg and "apply" in msg:
            return "application_help"
        if "what documents" in msg:
            return "documents"

        return "scheme_recommendation"

    def _contains_freshness_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in FRESHNESS_KEYWORDS)

    def _contains_detail_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in DETAIL_KEYWORDS)

    def _extract_primary_need(self, text: str) -> str:
        lowered = (text or "").lower()
        for need, keys in NEED_KEYWORDS.items():
            if any(k in lowered for k in keys):
                return need
        return ""

    def _should_use_web(
        self,
        user_query: str,
        extraction: Dict[str, Any],
        rag_evidence: List[Dict[str, str | float]],
    ) -> bool:
        if bool(extraction.get("requires_latest", False)):
            return True
        if self._contains_freshness_intent(user_query):
            return True
        return False

    def _extract_with_rules(self, user_query: str, user_profile: Dict[str, str] = None) -> Dict[str, object]:
        text = user_query.lower()
        missing = get_missing_required_fields(user_profile) if user_profile else []
        entities: Dict[str, str] = {
            "name": "",
            "occupation": "",
            "gender": "",
            "income": "",
            "loan_amount": "",
            "state": "",
        }

        # Name heuristic
        name_match = re.search(r"(?:my name is|i am) ([a-z\s]+)", text)
        if name_match:
            val = name_match.group(1).strip()
            if (
                len(val.split()) <= 3
                and val not in GREETING_HINTS
                and "farmer" not in val
                and "student" not in val
                and "labor" not in val
                and "unemployed" not in val
            ):
                entities["name"] = val.title()
        elif len(text.split()) <= 3 and text.strip() not in GREETING_HINTS and "skip" not in text:
            # Strictly map isolated short answers to whatever field we just asked for
            if missing:
                current_target = missing[0]
                if current_target == "name":
                    entities["name"] = text.title()
                elif current_target == "loan_amount":
                    entities["loan_amount"] = text
                elif current_target == "income":
                    entities["income"] = text
                elif current_target == "state":
                    norm_state = self._normalize_state(text)
                    entities["state"] = self._pretty_state(norm_state) if norm_state else text.title()
                elif current_target == "occupation":
                    entities["occupation"] = text
            else:
                entities["name"] = text.title()

        for occ, hints in OCCUPATION_HINTS.items():
            if any(h in text for h in hints):
                entities["occupation"] = occ
                break

        if re.search(r"\b(male|man|boy|gentleman)\b", text):
            entities["gender"] = "male"
        elif re.search(r"\b(female|woman|girl|lady)\b", text):
            entities["gender"] = "female"

        loan_match = re.search(r"loan\s*(?:of|for|amount)?\s*(\d+\s*lakhs?|\d+\s*k|\d+)", text)
        if loan_match:
            entities["loan_amount"] = loan_match.group(1)

        inc_specific = re.search(r"income\s*(?:is|of|range)?\s*([a-z\d\s-]+lakh)", text)
        if inc_specific:
            entities["income"] = inc_specific.group(1)
        else:
            income_match = re.search(
                r"(below\s*\d+\s*lakh|\d+\s*[-to]{1,3}\s*\d+\s*lakh|above\s*\d+\s*lakh|\d+\s*lakh)",
                text,
            )
            if income_match and not loan_match:
                entities["income"] = income_match.group(1)

        normalized_text = self._normalize_state(text)
        for state in STATE_KEYWORDS:
            if state in text or state == normalized_text:
                entities["state"] = state.title()
                break

        intent = "scheme_discovery"
        if "loan" in text:
            intent = "loan_support"
        elif "pension" in text:
            intent = "pension_support"
        elif "scholarship" in text:
            intent = "scholarship_support"

        return {
            "intent": intent,
            "entities": entities,
            "requires_latest": self._contains_freshness_intent(user_query),
        }

    def extract_intent_and_entities(
        self,
        user_query: str,
        user_profile: Dict[str, str],
    ) -> Dict[str, Any]:
        fallback = self._extract_with_rules(user_query, user_profile)
        if self.model is None:
            return fallback

        missing = get_missing_required_fields(user_profile)
        asked_for = missing[0] if missing else 'nothing specific'

        prompt = f"""
You are an information extraction engine.
Extract intent and profile entities from the user query.
Return valid JSON only, no markdown.
The user is currently being asked to provide: {asked_for}. Use this context to interpret short answers.
Do NOT put numerical amounts (like "1lakh" or "50000") into the "name" field! If it's a number, it belongs in income or loan_amount.
If the user's message is a single word answering the prompt for their name, extract it into the "name" field.
If the user specifies they do not know or do not have a certain detail (like "I don't know my income", "null", "none", "skip"), set that specific field's value exactly to "Not Specified" instead of leaving it empty.

Expected JSON schema:
{{
  "intent": "string",
  "requires_latest": true/false,
  "entities": {{
    "name": "string",
    "occupation": "string",
    "gender": "string",
    "income": "string",
    "loan_amount": "string",
    "state": "string"
  }}
}}

Current profile:
{json.dumps(user_profile, ensure_ascii=True)}

User query:
{user_query}
"""
        try:
            response = self.model.generate_content(prompt)
            text = (response.text or "").strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return fallback
            parsed = json.loads(match.group(0))
            entities = parsed.get("entities", {}) if isinstance(parsed, dict) else {}
            normalized = {
                "name": str(entities.get("name", "")).strip(),
                "occupation": str(entities.get("occupation", "")).strip(),
                "gender": str(entities.get("gender", "")).strip().lower(),
                "income": str(entities.get("income", "")).strip(),
                "loan_amount": str(entities.get("loan_amount", "")).strip(),
                "state": str(entities.get("state", "")).strip(),
            }
            return {
                "intent": str(parsed.get("intent", fallback["intent"])),
                "requires_latest": bool(
                    parsed.get("requires_latest", fallback["requires_latest"])
                ),
                "entities": normalized,
            }
        except Exception:
            return fallback

    def _merge_profile(self, user_profile: Dict[str, str], entities: Dict[str, str]) -> Dict[str, str]:
        updated = user_profile.copy()
        for key in ["name", "occupation", "gender", "income", "loan_amount", "state"]:
            value = str(entities.get(key, "")).strip()
            # Lock the profile mathematically so AI cannot overwrite an existing answer
            if value and not updated.get(key):
                if key == "state":
                    norm_state = self._normalize_state(value)
                    updated[key] = self._pretty_state(norm_state) if norm_state else value
                    continue
                if key == "gender":
                    updated[key] = value.lower()
                    continue
                if key == "occupation":
                    updated[key] = self._normalize_occupation(value)
                    continue
                updated[key] = value
        return updated

    def _infer_scheme_gender_target(self, scheme: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(scheme.get("name", "")),
                str(scheme.get("eligibility", "")),
                str(scheme.get("benefits", "")),
                str(scheme.get("application", "")),
            ]
        ).lower()
        women_patterns = [
            r"\bwomen\b",
            r"\bwoman\b",
            r"\bfemale\b",
            r"\bgirl\b",
            r"\bladies\b",
            r"\bfor women\b",
            r"\bwomen vendors\b",
        ]
        men_patterns = [
            r"\bmen\b",
            r"\bmale\b",
            r"\bfor men\b",
        ]
        women_hit = any(re.search(p, text) for p in women_patterns)
        men_hit = any(re.search(p, text) for p in men_patterns)
        if women_hit and not men_hit:
            return "female"
        if men_hit and not women_hit:
            return "male"
        return "any"

    def _format_evidence(
        self,
        rag_evidence: List[Dict[str, str | float]],
        web_evidence: List[Dict[str, str]],
        web_status: Dict[str, str] | None = None,
    ) -> str:
        lines: List[str] = []
        for item in rag_evidence:
            quote = str(item.get("quote", "")).strip()
            source = str(item.get("source", "unknown"))
            tags = str(item.get("tags", "")).strip()
            tag_suffix = f" [tags: {tags}]" if tags else ""
            lines.append(f'- RAG: "{quote}" (source: {source}){tag_suffix}')

        for item in web_evidence:
            title = item.get("title", "Official Update")
            url = item.get("url", "")
            domain = item.get("domain", "")
            lines.append(f"- Web: [{title}]({url}) ({domain})")

        if web_status and web_status.get("status") not in {"ok", "ready"}:
            lines.append(f"- Web Search Status: {web_status.get('detail', 'Unavailable.')}")

        if not lines:
            return "- No trustworthy evidence found from indexed documents or official web sources."
        return "\n".join(lines)

    def _build_web_query(
        self,
        user_query: str,
        extraction: Dict[str, Any],
        rag_evidence: List[Dict[str, str | float]],
        profile: Dict[str, str],
    ) -> str:
        scheme_names = [
            str(item.get("scheme_name", "")).strip()
            for item in rag_evidence[:2]
            if str(item.get("scheme_name", "")).strip()
        ]
        if scheme_names:
            return f"{' '.join(scheme_names)} official eligibility update site:gov.in"

        keywords = [
            str(extraction.get("intent", "")),
            profile.get("occupation", ""),
            profile.get("state", ""),
            user_query,
        ]
        compact = " ".join(part for part in keywords if part).replace("_", " ")
        return f"{compact} official government scheme site:gov.in"

    def _parse_scheme_cards(self, rag_evidence: List[Dict[str, str | float]]) -> List[Dict[str, str]]:
        cards: List[Dict[str, str]] = []
        seen_names = set()

        def extract_from_text(text: str, label: str) -> str:
            pattern = rf"{label}\s*:\s*(.+)"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = re.sub(r"\s+", " ", match.group(1).strip())
                return value

            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if s.strip()
            ]
            if not sentences:
                return "Not clearly specified in retrieved text."

            if label.lower() == "eligibility":
                for s in sentences:
                    if any(k in s.lower() for k in ["eligible", "eligibility", "beneficiary", "farmer", "applicant", "student", "sc", "st", "obc"]):
                        return s
                return sentences[0]

            if label.lower() == "benefits":
                for s in sentences:
                    if any(k in s.lower() for k in ["benefit", "support", "loan", "subsidy", "insurance", "credit", "pension", "assistance"]):
                        return s
                return sentences[min(1, len(sentences) - 1)]

            if label.lower() == "application steps":
                for s in sentences:
                    if any(k in s.lower() for k in ["apply", "application", "register", "visit", "contact", "portal", "bank"]):
                        return s
                return sentences[-1]

            return "Not clearly specified in retrieved text."

        for item in rag_evidence:
            quote = str(item.get("quote", "")).strip()
            if not quote:
                continue
            raw_name = str(item.get("scheme_name", "")).strip()
            source = str(item.get("source", "unknown"))

            name = re.sub(r"^\d+\.\s*", "", raw_name).strip(" *#")
            if not name or name.lower() == "unknown":
                first_line = quote.splitlines()[0].strip() if quote.splitlines() else ""
                name = first_line[:80] if first_line else "Unknown Scheme"

            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)

            link = extract_from_text(quote, "Official Link")
            if link == "Not clearly specified in retrieved text.":
                m = re.search(r'https?://[^\s]+', quote)
                link = m.group(0) if m else "Link not explicitly found"

            cards.append(
                {
                    "name": name,
                    "eligibility": extract_from_text(quote, "Eligibility"),
                    "benefits": extract_from_text(quote, "Benefits"),
                    "application": extract_from_text(quote, "Application Steps"),
                    "link": link,
                    "source": source,
                }
            )

        return cards

    def _to_amount(self, raw: str) -> int | None:
        text = (raw or "").lower().strip().replace(",", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|k)?", text)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2) or ""
        if unit in {"lakh", "lakhs"}:
            value *= 100000
        elif unit == "k":
            value *= 1000
        return int(value)

    def _extract_income_limit(self, text: str) -> int | None:
        lowered = (text or "").lower()
        patterns = [
            r"income[^.\n]{0,40}below\s*₹?\s*([\d,.]+)\s*(lakh|lakhs|k)?",
            r"income[^.\n]{0,40}up to\s*₹?\s*([\d,.]+)\s*(lakh|lakhs|k)?",
            r"income[^.\n]{0,40}less than\s*₹?\s*([\d,.]+)\s*(lakh|lakhs|k)?",
        ]
        for pattern in patterns:
            m = re.search(pattern, lowered)
            if not m:
                continue
            return self._to_amount(f"{m.group(1)} {m.group(2) or ''}")
        return None

    def _infer_scheme_states(self, scheme: Dict[str, Any]) -> List[str]:
        blob = " ".join(
            [
                str(scheme.get("name", "")),
                str(scheme.get("eligibility", "")),
                str(scheme.get("benefits", "")),
                str(scheme.get("application", "")),
            ]
        ).lower()
        states: List[str] = []
        for state in STATE_KEYWORDS:
            if state in blob:
                states.append(state)
        return states

    def generate_match_reasons(self, user: Dict[str, str], scheme: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        states = self._infer_scheme_states(scheme)

        profession = (user.get("occupation", "") or "").lower()
        if "farmer" in profession or "farming" in profession:
            reasons.append("You are a farmer ✔")

        user_state = (user.get("state", "") or "").lower().strip()
        if user_state and user_state in states:
            reasons.append(f"Scheme available in {user.get('state', '').title()} ✔")

        user_income = self._to_amount(user.get("income", ""))
        scheme_limit = self._extract_income_limit(
            " ".join(
                [
                    str(scheme.get("name", "")),
                    str(scheme.get("eligibility", "")),
                    str(scheme.get("benefits", "")),
                    str(scheme.get("application", "")),
                ]
            )
        )
        if user_income is not None and (scheme_limit is None or user_income <= scheme_limit):
            reasons.append("Your income fits eligibility ✔")

        if (user.get("loan_amount") or "").strip():
            reasons.append("Matches your financial need ✔")

        if not reasons:
            reasons.append("This looks relevant to your profile ✔")
        return reasons[:3]

    def rank_schemes(self, user: Dict[str, str], schemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        user_state = self._normalize_state(user.get("state", ""))
        primary_need = str(self.session.get("primary_need", "")).strip().lower()
        profession = self._normalize_occupation(user.get("occupation", ""))
        user_gender = (user.get("gender", "") or "").strip().lower()
        for scheme in schemes:
            score = 0
            text = " ".join(
                [
                    str(scheme.get("name", "")),
                    str(scheme.get("eligibility", "")),
                    str(scheme.get("benefits", "")),
                    str(scheme.get("application", "")),
                ]
            ).lower()
            scheme_states = self._infer_scheme_states(scheme)
            user_income = self._to_amount(user.get("income", ""))
            scheme_limit = self._extract_income_limit(text)
            scheme_gender_target = self._infer_scheme_gender_target(scheme)
            is_agri_scheme = any(
                k in text for k in ["farmer", "farming", "agriculture", "agri", "kisan", "crop", "horticulture", "livestock"]
            )

            # Hard-filter clear state mismatch when scheme explicitly targets other states.
            if user_state and scheme_states and user_state not in scheme_states:
                continue
            # Hard-filter gender-restricted schemes when user gender is missing or mismatched.
            if scheme_gender_target in {"male", "female"} and (
                user_gender not in {"male", "female"} or user_gender != scheme_gender_target
            ):
                continue
            # Domain hard-filter: farmer seeking loan should only see agri/farmer schemes.
            if profession == "farmer" and primary_need == "loan_support" and not is_agri_scheme:
                continue
            if user_state and user_state in text:
                score += 40
            if profession and profession in text:
                score += 30
            if profession == "farmer" and is_agri_scheme:
                score += 45
            if user_income is not None and scheme_limit is not None and user_income <= scheme_limit:
                score += 30
            if primary_need == "loan_support":
                if any(k in text for k in ["loan", "credit", "financial", "bank", "subsidy"]):
                    score += 40
                else:
                    score -= 20
            elif primary_need == "scholarship_support":
                if any(k in text for k in ["scholarship", "student", "education"]):
                    score += 40
                else:
                    score -= 20
            elif primary_need == "pension_support":
                if "pension" in text:
                    score += 40
                else:
                    score -= 20
            elif primary_need == "housing_support":
                if any(k in text for k in ["housing", "house", "home", "shelter", "awas"]):
                    score += 40
                else:
                    score -= 20

            item = dict(scheme)
            item["score"] = score
            item["why_match"] = self.generate_match_reasons(user, item)
            ranked.append(item)

        return sorted(ranked, key=lambda x: int(x.get("score", 0)), reverse=True)

    def _split_points(self, value: str) -> List[str]:
        text = (value or "").strip()
        if not text:
            return []
        parts = re.split(r"[;\n]|(?:\.\s+)", text)
        clean = [p.strip(" -•") for p in parts if p.strip()]
        return clean[:3] if clean else [text]

    def format_scheme_response(self, user: Dict[str, str], schemes: List[Dict[str, Any]]) -> str:
        prefix = "Got it 👍\n\n" if self.session.get("current_step") != STATES["RECOMMENDING"] else ""
        response = prefix + "Let me find the best options for you...\n\n"
        profession = user.get("occupation", "not specified")
        state = user.get("state", "not specified")
        income = user.get("income", "not specified")
        response += f"Based on your profile ({profession}, {state}, ₹{income} income):\n\n"
        response += "🎯 Top 3 schemes based on your profile:\n\n"

        for i, scheme in enumerate(schemes[:3]):
            tag = "⭐ Best Match" if i == 0 else f"{i+1}️⃣ Option"
            name = scheme.get("name", "Scheme")
            reasons = self.generate_match_reasons(user, scheme)
            benefits = self._split_points(str(scheme.get("benefits", "")))
            link = str(scheme.get("link", "")).strip() or "Official link not available"

            response += f"{tag}\n"
            response += f"🌾 *{name}*\n"
            response += "✔ Likely eligible based on your profile\n\n"
            response += "💰 Benefits:\n"
            for b in benefits[:2]:
                response += f"• {b}\n"

            response += "\n📋 Why this fits YOU:\n"
            for reason in reasons[:3]:
                response += f"• {reason}\n"

            response += f"\n🔗 Apply Now: {link}\n"
            response += "\n──────────────\n\n"

        response += "\nWhat would you like to do next?\n"
        response += "👉 Check required documents\n"
        response += "👉 See more schemes\n"
        response += "👉 Ask another question\n\n"
        response += "⚠ Always verify on the official government website."
        return response

    def handle_documents(self, user: Dict[str, str], scheme: Dict[str, Any] | None) -> str:
        if not scheme:
            return "Let me first find a suitable scheme for you 👍"

        response = "Got it 👍 Here's what you'll need for this scheme:\n\n"
        response += "📄 Documents:\n"
        docs = scheme.get(
            "documents",
            [
                "Aadhaar Card",
                "Bank Account Details",
                "Land Ownership Proof",
                "Income Certificate",
            ],
        )
        for d in docs:
            response += f"• {d}\n"

        response += "\n🪜 Next steps:\n"
        response += "• Gather these documents\n"
        response += "• Visit the official website\n"
        response += "• Submit application form\n"

        response += f"\n🔗 Apply here:\n{scheme.get('link', 'Official link not available')}"
        response += "\n\n⚠ Make sure all documents are valid.\n"
        response += "\nWant help with application steps?"
        return response

    def handle_application_steps(self, scheme: Dict[str, Any] | None) -> str:
        if not scheme:
            return "Let me first find a suitable scheme for you 👍"
        return (
            "🪜 Here's how to apply:\n\n"
            "1️⃣ Visit the official website\n"
            "2️⃣ Register / Login\n"
            "3️⃣ Fill application form\n"
            "4️⃣ Upload required documents\n"
            "5️⃣ Submit and track status\n\n"
            f"🔗 Apply here:\n{scheme.get('link', 'Official link not available')}\n\n"
            "Need help with any step? 👍"
        )

    def handle_eligibility_requirements(
        self,
        user: Dict[str, str],
        scheme: Dict[str, Any] | None,
    ) -> str:
        if not scheme:
            return "Let me first find a suitable scheme for you 👍"

        scheme_name = str(scheme.get("name", "this scheme"))
        eligibility_text = str(scheme.get("eligibility", "")).strip()
        lines: List[str] = []
        if eligibility_text:
            # Split on sentence/line boundaries only (not individual letters).
            lines = [seg.strip(" -•") for seg in re.split(r"[.;\n]+", eligibility_text) if seg.strip()]

        response = f"Got it 👍\n\n📋 Eligibility requirements for {scheme_name}:\n"
        if lines:
            for item in lines[:4]:
                response += f"• {item}\n"
        else:
            response += "• Eligibility depends on profession/state/income criteria\n"
            response += "• Check category-specific conditions if applicable\n"

        response += "\n🧾 Your profile snapshot:\n"
        response += f"• Profession: {user.get('occupation', 'Not provided')}\n"
        response += f"• State: {user.get('state', 'Not provided')}\n"
        response += f"• Income: {user.get('income', 'Not provided')}\n"
        response += "\n🔗 Official link:\n"
        response += f"{scheme.get('link', 'Official link not available')}\n"
        response += "\n⚠ Always verify final eligibility on the official website."
        return response

    def handle_more_schemes(
        self,
        user: Dict[str, str],
        session: Dict[str, Any],
        all_schemes: List[Dict[str, Any]],
    ) -> str:
        if not all_schemes:
            return "Let me first find a suitable scheme for you 👍"
        shown = session.get("last_schemes", [])
        shown_names = {str(x.get("name", "")).strip().lower() for x in shown}
        remaining = [s for s in all_schemes if str(s.get("name", "")).strip().lower() not in shown_names]

        if not remaining:
            return "You've already seen the best matching schemes 👍 Want help applying?"

        new_schemes = remaining[:3]
        session["last_schemes"] = shown + new_schemes
        return self.format_scheme_response(user, new_schemes)

    def _card_relevance(self, card: Dict[str, str], query: str, profile: Dict[str, str]) -> int:
        bag = " ".join(
            [
                card.get("name", ""),
                card.get("eligibility", ""),
                card.get("benefits", ""),
                card.get("application", ""),
            ]
        ).lower()
        query_terms = re.findall(r"[a-zA-Z]{3,}", query.lower())
        score = sum(1 for t in query_terms if t in bag)

        occ = (profile.get("occupation") or "").lower()
        if occ and occ in bag:
            score += 3
        cat = (profile.get("category") or "").lower()
        if cat and cat in bag:
            score += 2
        if "loan" in query.lower() and any(k in bag for k in ["loan", "credit", "bank"]):
            score += 2
        elif "loan" in query.lower():
            score -= 2
        if "pension" in query.lower() and "pension" in bag:
            score += 2
        elif "pension" in query.lower():
            score -= 3
        if "scholarship" in query.lower() and "scholarship" in bag:
            score += 2
        elif "scholarship" in query.lower():
            score -= 2
        if any(term in query.lower() for term in ["housing", "house", "home", "rural"]):
            if any(k in bag for k in ["housing", "house", "awas", "rural", "pucca"]):
                score += 2
            else:
                score -= 2
        return score

    def _score_match_reason(
        self,
        scheme: Dict[str, str],
        query: str,
        profile: Dict[str, str],
    ) -> str:
        text = " ".join(
            [
                scheme.get("name", ""),
                scheme.get("eligibility", ""),
                scheme.get("benefits", ""),
                scheme.get("application", ""),
            ]
        ).lower()
        reasons: List[str] = []
        occ = (profile.get("occupation") or "").lower()
        if occ and occ in {"farmer", "laborer", "self-employed", "student"}:
            if occ == "farmer" and any(x in text for x in ["farmer", "kisan", "krishi", "crop"]):
                reasons.append("occupation alignment")
            if occ == "student" and "scholarship" in text:
                reasons.append("education support alignment")
            if occ == "self-employed" and any(x in text for x in ["loan", "credit", "enterprise", "msme"]):
                reasons.append("business/credit alignment")
            if occ == "laborer" and any(x in text for x in ["worker", "livelihood", "employment"]):
                reasons.append("livelihood alignment")

        q = query.lower()
        if "loan" in q and any(x in text for x in ["loan", "credit", "subvention"]):
            reasons.append("loan intent match")
        if "pension" in q and "pension" in text:
            reasons.append("pension intent match")

        category = (profile.get("category") or "").lower()
        if category and category in text:
            reasons.append("category mention present")

        if not reasons:
            return "Possible fit based on partial evidence; verify exact criteria on official portal."
        return "Likely fit due to " + ", ".join(reasons) + "."

    def _detailed_fallback_markdown(
        self,
        query: str,
        profile: Dict[str, str],
        rag_evidence: List[Dict[str, str | float]],
        web_evidence: List[Dict[str, str]],
        used_web: bool,
        web_status: Dict[str, str] | None = None,
    ) -> str:
        cards = self._parse_scheme_cards(rag_evidence)
        cards = self.rank_schemes(profile, cards)
        cards = cards[:3]

        if not cards and not rag_evidence and not web_evidence:
            return (
                "Got it 👍\n\n"
                "I could not find a strong scheme match yet.\n"
                "• Try sharing your state, profession, and income range\n"
                "• I can then rank top schemes for you\n\n"
                "⚠ Always verify details on the official government website.\n"
            )
        return self.format_scheme_response(profile, cards)

    def _fallback_markdown(
        self,
        query: str,
        profile: Dict[str, str],
        rag_evidence: List[Dict[str, str | float]],
        web_evidence: List[Dict[str, str]],
        web_status: Dict[str, str] | None = None,
    ) -> str:
        if not rag_evidence and not web_evidence:
            return (
                "Got it 👍\n\n"
                "I could not find a reliable match right now.\n"
                "• Please share more details so I can improve matching\n"
                "• State + profession + income usually helps most\n\n"
                "⚠ Always verify details on the official government website.\n"
            )
        cards = self.rank_schemes(profile, self._parse_scheme_cards(rag_evidence))
        if cards:
            return self.format_scheme_response(profile, cards[:3])
        return (
            "Got it 👍\n\n"
            "I found information, but not enough to rank cleanly.\n"
            "• Ask me to show more schemes\n"
            "• Or share one more detail for better matching\n\n"
            "⚠ Always verify details on the official government website.\n"
        )

    def _synthesize_markdown(
        self,
        query: str,
        profile: Dict[str, str],
        rag_evidence: List[Dict[str, str]],
        web_evidence: List[Dict[str, str]],
        used_web: bool,
        web_status: Dict[str, str] = None,
        chat_history: List[Dict[str, str]] = None,
    ) -> Any:
        if not rag_evidence or len(rag_evidence) == 0:
            def empty_generator():
                yield "I couldn't find relevant information in the database."
            return empty_generator()

        if self.model is None:
            return self._detailed_fallback_markdown(
                query,
                profile,
                rag_evidence,
                web_evidence,
                used_web=used_web,
                web_status=web_status,
            )

        formatted_context = "\n\n".join([
            f"Source {i+1}:\n{doc.get('quote', '')}\nLink: {doc.get('source', '')}" 
            for i, doc in enumerate(rag_evidence)
        ])

        prompt = f"""
{SYSTEM_PROMPT}

Answer ONLY from the context below. Do not invent scheme names, numbers, or benefits.
If the context does not contain the answer, say "I couldn't find relevant information in the database."

Context:
{formatted_context}

Question:
{query}

Recent History:
{json.dumps(chat_history[-4:] if chat_history else [], ensure_ascii=True)}
"""
        try:
            response = self.model.generate_content(prompt)
            text = (response.text or "").strip()
            if text:
                if "⚠ Always verify details on the official government website." not in text:
                    text += "\n\n⚠ Always verify details on the official government website."
                if "What would you like to do next?" not in text:
                    text += (
                        "\n\nWhat would you like to do next?\n"
                        "• Check documents needed\n"
                        "• See more schemes\n"
                        "• Ask another question\n"
                    )
                return text
            return self._detailed_fallback_markdown(
                query,
                profile,
                rag_evidence,
                web_evidence,
                used_web=used_web,
                web_status=web_status,
            )
        except Exception:
            return self._detailed_fallback_markdown(
                query,
                profile,
                rag_evidence,
                web_evidence,
                used_web=used_web,
                web_status=web_status,
            )

    def handle_query(
        self,
        user_query: Any,
        user_profile: Dict[str, str],
        chat_history: List[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        query_text = ""
        if isinstance(user_query, dict):
            query_text = str(user_query.get("text", "")).strip()
        else:
            query_text = str(user_query or "").strip()

        if self._is_greeting_text(query_text):
            self.session["last_intent"] = "greeting"
            self.session["current_step"] = None
            return {
                "updated_profile": user_profile,
                "response_markdown": "Hello this is Jan-Sahayak AI, what do you need for today? I can help you with that.",
                "route": "greeting",
            }

        inferred_need = self._extract_primary_need(query_text)
        if inferred_need:
            self.session["primary_need"] = inferred_need

        intent = self.detect_intent(user_query, self.session)
        # Safety guard: do not trigger fresh RAG for non-recommendation follow-ups.
        skip_rag = intent != "scheme_recommendation"
        if skip_rag:
            if intent == "documents":
                response = self.handle_documents(user_profile, self.session.get("last_scheme"))
                self.session["last_intent"] = intent
                self.session["current_step"] = STATES["SHOWING_DOCS"]
                return {
                    "updated_profile": user_profile,
                    "response_markdown": response,
                    "route": "documents",
                }
            if intent == "eligibility_help":
                response = self.handle_eligibility_requirements(
                    user_profile,
                    self.session.get("last_scheme"),
                )
                self.session["last_intent"] = intent
                self.session["current_step"] = STATES["SHOWING_DOCS"]
                return {
                    "updated_profile": user_profile,
                    "response_markdown": response,
                    "route": "eligibility_help",
                }
            if intent == "more_schemes":
                response = self.handle_more_schemes(
                    user_profile,
                    self.session,
                    self.last_ranked_schemes,
                )
                self.session["last_intent"] = intent
                self.session["current_step"] = STATES["RECOMMENDING"]
                return {
                    "updated_profile": user_profile,
                    "response_markdown": response,
                    "route": "more_schemes",
                }
            if intent == "application_help":
                response = self.handle_application_steps(self.session.get("last_scheme"))
                self.session["last_intent"] = intent
                self.session["current_step"] = STATES["APPLICATION_HELP"]
                return {
                    "updated_profile": user_profile,
                    "response_markdown": response,
                    "route": "application_help",
                }
            if intent == "completed":
                self.session["last_intent"] = intent
                self.session["current_step"] = None
                return {
                    "updated_profile": user_profile,
                    "response_markdown": "You're all set 👍 Let me know if you need anything else!",
                    "route": "completed",
                }
            if intent == "new_query":
                self.session = {
                    "last_intent": None,
                    "last_scheme": None,
                    "last_schemes": [],
                    "current_step": None,
                    "primary_need": "",
                }
                self.last_scheme = None
                self.last_ranked_schemes = []
                return {
                    "updated_profile": user_profile,
                    "response_markdown": "Sure 👍 Tell me what you need help with!",
                    "route": "new_query",
                }

        extraction = self.extract_intent_and_entities(query_text, user_profile)
        entities = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
        updated_profile = self._merge_profile(user_profile, entities if isinstance(entities, dict) else {})

        if self.session.get("current_step") in [STATES["SHOWING_DOCS"], STATES["APPLICATION_HELP"]]:
            if intent == "scheme_recommendation":
                self.session["last_intent"] = intent
                return {
                    "updated_profile": updated_profile,
                    "response_markdown": "Do you want to continue with this scheme or explore others? 👍",
                    "route": "flow_guard",
                }

        missing_fields = get_missing_required_fields(updated_profile)
        if missing_fields:
            followup = next_followup_question(missing_fields, updated_profile)
            response_text = followup
            self.session["last_intent"] = "followup"
            return {
                "updated_profile": updated_profile,
                "response_markdown": response_text,
                "route": "followup",
            }

        # Contextual Query Expansion (RAG From Scratch Technique)
        search_query = query_text
        if len(query_text.split()) <= 4:
            parts = []
            for k in ["occupation", "income", "state"]:
                val = updated_profile.get(k)
                if val and val.lower() not in ["not specified", "unknown", "skip"]:
                    parts.append(val)
            need_hint = str(self.session.get("primary_need", "")).replace("_", " ")
            if need_hint:
                parts.append(need_hint)
            if parts:
                search_query = f"{query_text} for {' '.join(parts)}"

        # Metadata Exact Filtering
        state_filter = updated_profile.get("state", "").strip() or None
        if state_filter and state_filter.lower() in ["not specified", "unknown", "skip"]:
            state_filter = None
        if state_filter:
            state_filter = self._pretty_state(self._normalize_state(state_filter))

        rag_evidence = self.rag_agent.retrieve(search_query, top_k=self.top_k_rag, state_filter=state_filter)
        use_web = self._should_use_web(query_text, extraction, rag_evidence)
        web_evidence: List[Dict[str, str]] = []
        web_status: Dict[str, str] = {"status": "not_used", "detail": "Web search was not used."}
        if use_web:
            web_query = self._build_web_query(query_text, extraction, rag_evidence, updated_profile)
            web_evidence = self.web_agent.search_official(web_query, max_results=self.top_k_web)
            web_status = getattr(self.web_agent, "last_status", web_status)

        markdown = self._synthesize_markdown(
            query_text,
            updated_profile,
            rag_evidence,
            web_evidence,
            use_web,
            web_status=web_status,
            chat_history=chat_history,
        )

        # Save memory for follow-up intents like documents / apply / more schemes.
        ranked_now = self.rank_schemes(updated_profile, self._parse_scheme_cards(rag_evidence))
        self.last_ranked_schemes = ranked_now[:]
        self.last_scheme = ranked_now[0] if ranked_now else None
        self.session["last_schemes"] = ranked_now[:3]
        self.session["last_scheme"] = ranked_now[0] if ranked_now else None
        self.session["last_intent"] = intent
        self.session["current_step"] = STATES["RECOMMENDING"]

        return {
            "updated_profile": updated_profile,
            "response_markdown": markdown,
            "route": "rag_web" if use_web else "rag_only",
            "rag_evidence": rag_evidence,
            "web_evidence": web_evidence,
            "web_status": web_status,
        }
