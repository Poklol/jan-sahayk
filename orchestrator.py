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

    def _contains_freshness_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in FRESHNESS_KEYWORDS)

    def _contains_detail_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in DETAIL_KEYWORDS)

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
            "income": "",
            "loan_amount": "",
            "state": "",
        }

        # Name heuristic
        name_match = re.search(r"(?:my name is|i am) ([a-z\s]+)", text)
        if name_match:
            val = name_match.group(1).strip()
            if len(val.split()) <= 3 and "farmer" not in val and "student" not in val and "labor" not in val and "unemployed" not in val:
                entities["name"] = val.title()
        elif len(text.split()) <= 3 and "hello" not in text and "hi" not in text and "skip" not in text:
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
                    entities["state"] = text.title()
                elif current_target == "occupation":
                    entities["occupation"] = text
            else:
                entities["name"] = text.title()

        for occ, hints in OCCUPATION_HINTS.items():
            if any(h in text for h in hints):
                entities["occupation"] = occ
                break

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

        for state in STATE_KEYWORDS:
            if state in text:
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
        for key in ["name", "occupation", "income", "loan_amount", "state"]:
            value = str(entities.get(key, "")).strip()
            # Lock the profile mathematically so AI cannot overwrite an existing answer
            if value and not updated.get(key):
                updated[key] = value
        return updated

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
        cards.sort(key=lambda c: self._card_relevance(c, query, profile), reverse=True)
        scored = [(card, self._card_relevance(card, query, profile)) for card in cards]
        strong = [card for card, score in scored if score >= 3]
        cards = strong[:4]

        if not cards and not rag_evidence and not web_evidence:
            return "I'm sorry, I couldn't find any reliable matching welfare schemes from our database or official web sources based on the details you provided. Could you try adjusting your request or providing more details?"

        response_parts = []
        if cards:
            top_names = ", ".join(card["name"] for card in cards[:3])
            response_parts.append(f"Based on your profile, I found some relevant schemes that might be a great fit for you! The most promising ones are {top_names}.\n")
        else:
            response_parts.append("I couldn't find a perfect match, but I did find some documents that might include partial information you'd find helpful.\n")
            
        for card in cards[:4]:
            url = card.get('link', 'Link not explicitly found')
                
            response_parts.append(f"- **Scheme Name**: {card['name']}")
            response_parts.append(f"  - **Eligibility**: {card['eligibility']}")
            response_parts.append(f"  - **Benefits**: {card['benefits']}")
            response_parts.append(f"  - **Official Link**: {url}\n")

        if used_web and web_evidence:
            response_parts.append("\nI also checked the web for some of the latest official updates related to your query:")
            for web in web_evidence[:3]:
                response_parts.append(f"- [{web.get('title', 'Official Link')}]({web.get('url', '')})")

        response_parts.append("\nPlease review these details and kindly visit the official portals linked above to verify the complete eligibility criteria!")
        return "\n".join(response_parts)

    def _fallback_markdown(
        self,
        query: str,
        profile: Dict[str, str],
        rag_evidence: List[Dict[str, str | float]],
        web_evidence: List[Dict[str, str]],
        web_status: Dict[str, str] | None = None,
    ) -> str:
        if not rag_evidence and not web_evidence:
            return "I couldn't find a reliable matching welfare scheme for your situation right now. It is possible that the exact combination of criteria doesn't match an active scheme. Could you try clarifying your goal?"

        answer = "I found some potentially relevant information based on your answers! I only surface the strongest matches as full recommendations, so please feel free to review the details below."
        return answer + "\n\n" + self._format_evidence(rag_evidence, web_evidence, web_status)

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
            response = self.model.generate_content(prompt, stream=True)
            
            def stream_generator():
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            return stream_generator()
        except Exception as e:
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
        user_query: str,
        user_profile: Dict[str, str],
        chat_history: List[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        if user_query.strip().lower() in ["hi", "hello", "hey", "greetings"]:
            return {
                "updated_profile": user_profile,
                "response_markdown": "Hello this is Jan-Sahayak AI, what do you need for today? I can help you with that.",
                "route": "greeting",
            }

        extraction = self.extract_intent_and_entities(user_query, user_profile)
        entities = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
        updated_profile = self._merge_profile(user_profile, entities if isinstance(entities, dict) else {})

        missing_fields = get_missing_required_fields(updated_profile)
        if missing_fields:
            followup = next_followup_question(missing_fields)
            response_text = followup
            return {
                "updated_profile": updated_profile,
                "response_markdown": response_text,
                "route": "followup",
            }

        # Contextual Query Expansion (RAG From Scratch Technique)
        search_query = user_query
        if len(user_query.split()) <= 4:
            parts = []
            for k in ["occupation", "income", "state"]:
                val = updated_profile.get(k)
                if val and val.lower() not in ["not specified", "unknown", "skip"]:
                    parts.append(val)
            if parts:
                search_query = f"{user_query} for {' '.join(parts)}"

        # Metadata Exact Filtering
        state_filter = updated_profile.get("state", "").strip() or None
        if state_filter and state_filter.lower() in ["not specified", "unknown", "skip"]:
            state_filter = None

        rag_evidence = self.rag_agent.retrieve(search_query, top_k=self.top_k_rag, state_filter=state_filter)
        use_web = self._should_use_web(user_query, extraction, rag_evidence)
        web_evidence: List[Dict[str, str]] = []
        web_status: Dict[str, str] = {"status": "not_used", "detail": "Web search was not used."}
        if use_web:
            web_query = self._build_web_query(user_query, extraction, rag_evidence, updated_profile)
            web_evidence = self.web_agent.search_official(web_query, max_results=self.top_k_web)
            web_status = getattr(self.web_agent, "last_status", web_status)

        markdown = self._synthesize_markdown(
            user_query,
            updated_profile,
            rag_evidence,
            web_evidence,
            use_web,
            web_status=web_status,
            chat_history=chat_history,
        )

        return {
            "updated_profile": updated_profile,
            "response_markdown": markdown,
            "route": "rag_web" if use_web else "rag_only",
            "rag_evidence": rag_evidence,
            "web_evidence": web_evidence,
            "web_status": web_status,
        }
