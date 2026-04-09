from __future__ import annotations

from typing import Dict, List

import streamlit as st

DEFAULT_PROFILE: Dict[str, str] = {
    "occupation": "",
    "income": "",
    "state": "",
    "category": "",
}


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = DEFAULT_PROFILE.copy()


def append_message(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def update_profile(profile_updates: Dict[str, str]) -> Dict[str, str]:
    profile = st.session_state.user_profile.copy()
    for key, value in profile_updates.items():
        if key in profile and isinstance(value, str) and value.strip():
            profile[key] = value.strip()
    st.session_state.user_profile = profile
    return profile


def get_missing_required_fields(profile: Dict[str, str]) -> List[str]:
    required = ["occupation", "income", "state"]
    return [field for field in required if not profile.get(field, "").strip()]


def next_followup_question(missing_fields: List[str]) -> str:
    question_map = {
        "occupation": "What is your occupation (for example: farmer, student, laborer, self-employed)?",
        "income": "What is your yearly family income range (for example: below 2 lakh, 2-5 lakh, above 5 lakh)?",
        "state": "Which state do you live in?",
        "category": "If relevant, what is your social category (SC/ST/OBC/General/EWS)?",
    }
    if not missing_fields:
        return "Could you share a bit more about your situation so I can check scheme eligibility?"
    return question_map.get(missing_fields[0], "Could you provide one more detail so I can continue?")


def profile_snapshot(profile: Dict[str, str]) -> str:
    lines = [
        f"- Occupation: {profile.get('occupation') or 'Not provided'}",
        f"- Income: {profile.get('income') or 'Not provided'}",
        f"- State: {profile.get('state') or 'Not provided'}",
        f"- Category: {profile.get('category') or 'Not provided'}",
    ]
    return "\n".join(lines)


def reset_state() -> None:
    st.session_state.chat_history = []
    st.session_state.user_profile = DEFAULT_PROFILE.copy()
