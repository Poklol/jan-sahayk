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

    def _extract_with_rules(self, user_query: str) -> Dict[str, object]:
        text = user_query.lower()
        entities: Dict[str, str] = {
            "occupation": "",
            "income": "",
            "state": "",
            "category": "",
        }

        for occ, hints in OCCUPATION_HINTS.items():
            if any(h in text for h in hints):
                entities["occupation"] = occ
                break

        income_match = re.search(
            r"(below\s*\d+\s*lakh|\d+\s*[-to]{1,3}\s*\d+\s*lakh|above\s*\d+\s*lakh|\d+\s*lakh)",
            text,
        )
        if income_match:
            entities["income"] = income_match.group(1)

        for state in STATE_KEYWORDS:
            if state in text:
                entities["state"] = state.title()
                break

        for category, hints in CATEGORY_HINTS.items():
            if any(h in text for h in hints):
                entities["category"] = category
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
        fallback = self._extract_with_rules(user_query)
        if self.model is None:
            return fallback

        prompt = f"""
You are an information extraction engine.
Extract intent and profile entities from the user query.
Return valid JSON only, no markdown.

Expected JSON schema:
{{
  "intent": "string",
  "requires_latest": true/false,
  "entities": {{
    "occupation": "string",
    "income": "string",
    "state": "string",
    "category": "string"
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
                "occupation": str(entities.get("occupation", "")).strip(),
                "income": str(entities.get("income", "")).strip(),
                "state": str(entities.get("state", "")).strip(),
                "category": str(entities.get("category", "")).strip(),
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
        for key in ["occupation", "income", "state", "category"]:
            value = str(entities.get(key, "")).strip()
            if value:
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

            cards.append(
                {
                    "name": name,
                    "eligibility": extract_from_text(quote, "Eligibility"),
                    "benefits": extract_from_text(quote, "Benefits"),
                    "application": extract_from_text(quote, "Application Steps"),
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
            return (
                "1. ✅ Final Answer\n"
                "I could not find a reliable matching welfare scheme from indexed documents and official web sources for this query.\n\n"
                "2. 📊 Eligibility Check\n"
                f"Profile considered: occupation={profile.get('occupation') or 'NA'}, income={profile.get('income') or 'NA'}, state={profile.get('state') or 'NA'}, category={profile.get('category') or 'NA'}. "
                "No verifiable scheme criteria were retrieved.\n\n"
                "3. 📚 Evidence\n"
                f"{self._format_evidence(rag_evidence, web_evidence, web_status)}\n\n"
                "4. 🧠 Reasoning\n"
                "The system attempted retrieval but did not get trustworthy content for a confident recommendation."
            )

        if cards:
            top_names = ", ".join(card["name"] for card in cards[:3])
            answer_line = (
                f"Based on your profile, the most relevant schemes appear to be: {top_names}. "
                "Details below are extracted from indexed scheme documents and should be verified on official portals."
            )
        else:
            answer_line = (
                "I did not find a strong scheme match for this request in the indexed documents. "
                "I’m showing only partial evidence below, and you should not treat it as a recommendation."
            )

        eligibility_lines = [
            f"- Profile considered: occupation={profile.get('occupation') or 'NA'}, income={profile.get('income') or 'NA'}, "
            f"state={profile.get('state') or 'NA'}, category={profile.get('category') or 'NA'}."
        ]
        for card in cards[:4]:
            eligibility_lines.append(f"- {card['name']}: {self._score_match_reason(card, query, profile)}")

        detail_lines: List[str] = []
        for card in cards[:4]:
            detail_lines.append(f"- Scheme: {card['name']}")
            detail_lines.append(f"  Eligibility: {card['eligibility']}")
            detail_lines.append(f"  Benefits: {card['benefits']}")
            detail_lines.append(f"  How to Apply: {card['application']}")
            detail_lines.append(f"  Source File: {card['source']}")

        evidence_lines = []
        if detail_lines:
            evidence_lines.extend(detail_lines)
        elif rag_evidence:
            evidence_lines.append("- RAG matched only weak/partial evidence, so no scheme was promoted as a recommendation.")
        for item in rag_evidence[:3]:
            quote = str(item.get("quote", "")).strip().replace("\n", " ")
            source = str(item.get("source", "unknown"))
            tags = str(item.get("tags", "")).strip()
            tag_suffix = f" [tags: {tags}]" if tags else ""
            evidence_lines.append(f"- RAG Quote: \"{quote[:350]}\" (source: {source}){tag_suffix}")
        if web_evidence:
            for web in web_evidence[:4]:
                evidence_lines.append(
                    f"- Web Source: [{web.get('title', 'Official Update')}]({web.get('url', '')}) ({web.get('domain', '')})"
                )
        elif web_status and web_status.get("status") not in {"ok", "ready"}:
            evidence_lines.append(f"- Web Search Status: {web_status.get('detail', 'Unavailable.')}")
        if not evidence_lines:
            evidence_lines.append("- No valid evidence found.")

        reasoning = (
            "User profile was extracted and matched against retrieved scheme content. "
            f"RAG returned {len(rag_evidence)} evidence chunk(s). "
            + (
                f"Official web search was used and returned {len(web_evidence)} filtered result(s)."
                if used_web and web_evidence
                else (
                    f"Official web search was attempted but did not produce usable results: {web_status.get('detail', 'Unavailable.')}"
                    if used_web
                    else "Web search was skipped because the query did not require freshness and RAG evidence was sufficient."
                )
            )
        )

        eligibility_text = "\n".join(eligibility_lines)
        evidence_text = "\n".join(evidence_lines)

        return (
            "1. ✅ Final Answer\n"
            f"{answer_line}\n\n"
            "2. 📊 Eligibility Check\n"
            f"{eligibility_text}\n\n"
            "3. 📚 Evidence\n"
            f"{evidence_text}\n\n"
            "4. 🧠 Reasoning\n"
            f"{reasoning}"
        )

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
                "1. ✅ Final Answer\n"
                "I could not find a reliable matching welfare scheme from the indexed documents or official web updates right now.\n\n"
                "2. 📊 Eligibility Check\n"
                f"Based on your profile ({profile}), there is not enough verified scheme evidence to confirm eligibility.\n\n"
                "3. 📚 Evidence\n"
                f"{self._format_evidence(rag_evidence, web_evidence, web_status)}\n\n"
                "4. 🧠 Reasoning\n"
                "Your request was analyzed, but no trusted scheme match was retrieved. Please refine your query with scheme type or benefit goal."
            )

        answer = "I found potentially relevant evidence. Only stronger matches are surfaced as recommendations, so please review the evidence and verify final eligibility on the official portal."
        return (
            "1. ✅ Final Answer\n"
            f"{answer}\n\n"
            "2. 📊 Eligibility Check\n"
            f"Profile considered: occupation={profile.get('occupation') or 'NA'}, income={profile.get('income') or 'NA'}, "
            f"state={profile.get('state') or 'NA'}, category={profile.get('category') or 'NA'}. "
            "Eligibility is a preliminary check pending exact scheme criteria.\n\n"
            "3. 📚 Evidence\n"
            f"{self._format_evidence(rag_evidence, web_evidence, web_status)}\n\n"
            "4. 🧠 Reasoning\n"
            "The assistant matched your profile and query intent to retrieved scheme snippets, then added official web updates when freshness was requested."
        )

    def _synthesize_markdown(
        self,
        query: str,
        profile: Dict[str, str],
        rag_evidence: List[Dict[str, str | float]],
        web_evidence: List[Dict[str, str]],
        used_web: bool,
        web_status: Dict[str, str] | None = None,
    ) -> str:
        if self.model is None:
            return self._detailed_fallback_markdown(
                query,
                profile,
                rag_evidence,
                web_evidence,
                used_web=used_web,
                web_status=web_status,
            )

        prompt = f"""
You are Jan-Sahayak AI, a welfare-scheme assistant.
Use only provided evidence.
If evidence is weak, say clearly that no reliable scheme was found.
Do not invent scheme names, numbers, or benefits.

Return markdown with exactly these 4 numbered sections:
1. ✅ Final Answer
2. 📊 Eligibility Check
3. 📚 Evidence
4. 🧠 Reasoning

Quality requirements:
- Be coherent and detailed.
- In Final Answer, mention 2-4 scheme names if available.
- In Eligibility Check, explain likely fit scheme by scheme.
- In Evidence, include scheme-wise details: eligibility, benefits, application steps, and source.
- In Reasoning, explicitly mention whether web search was used.

User Query:
{query}

User Profile:
{json.dumps(profile, ensure_ascii=True)}

RAG Evidence:
{json.dumps(rag_evidence, ensure_ascii=True)}

Web Evidence:
{json.dumps(web_evidence, ensure_ascii=True)}

Web Used: {used_web}
Web Status: {json.dumps(web_status or {}, ensure_ascii=True)}
"""
        try:
            response = self.model.generate_content(prompt)
            text = (response.text or "").strip()
            if "1. ✅ Final Answer" in text and "2. 📊 Eligibility Check" in text:
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
        user_query: str,
        user_profile: Dict[str, str],
    ) -> Dict[str, object]:
        extraction = self.extract_intent_and_entities(user_query, user_profile)
        entities = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
        updated_profile = self._merge_profile(user_profile, entities if isinstance(entities, dict) else {})

        missing_fields = get_missing_required_fields(updated_profile)
        if missing_fields:
            followup = next_followup_question(missing_fields)
            response_text = (
                "I can help with that. I need one more detail before checking schemes.\n\n"
                f"{followup}"
            )
            return {
                "updated_profile": updated_profile,
                "response_markdown": response_text,
                "route": "followup",
            }

        rag_evidence = self.rag_agent.retrieve(user_query, top_k=self.top_k_rag)
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
        )

        return {
            "updated_profile": updated_profile,
            "response_markdown": markdown,
            "route": "rag_web" if use_web else "rag_only",
            "rag_evidence": rag_evidence,
            "web_evidence": web_evidence,
            "web_status": web_status,
        }
