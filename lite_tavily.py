import os
import json
import re
from typing import Optional, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq


# ------------------------
# Load environment
# ------------------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TAVILY_API_KEY:
    st.error("Missing TAVILY_API_KEY in .env")
    st.stop()

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()

# ------------------------
# Clients
# ------------------------
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"


# ------------------------
# Helpers
# ------------------------
def _strip_code_fences(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    return t


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _build_search_query(raw: str) -> str:
    if "tata" not in raw.lower():
        return f"Tata Motors dealer showroom {raw}"
    return raw


def tavily_search(user_query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    res = tavily_client.search(
        query=user_query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )
    return res.get("results", []) if isinstance(res, dict) else []


def groq_structured_answer(user_query: str, web_results: List[Dict[str, Any]]) -> Dict[str, Any]:

    compact = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "content": (r.get("content") or "")[:1200],
        }
        for r in web_results
    ]

    system_prompt = """
You are a dealership-finder assistant for India.
You must return ONLY valid JSON (no markdown, no commentary).

Rules:
- Use ONLY the provided web results. Do NOT invent details.
- If address/phone/hours are not present in results, set them to null.
- Pick the top 3 most relevant Tata Motors dealers/showrooms.
- "confidence" must be one of: "high", "medium", "low".
- If no clear location, set follow_up_question.
Return JSON in specified schema.
"""

    user_prompt = f"""
User query: {user_query}

Web results:
{json.dumps(compact, ensure_ascii=False)}
"""

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content or ""
    text = _strip_code_fences(text)
    parsed = _safe_json_loads(text)

    if parsed is None:
        return {
            "mode": "text_fallback",
            "query": user_query,
            "response": text,
            "sources": compact,
        }

    parsed["mode"] = "structured"
    parsed["sources_used"] = [
        {"title": s.get("title"), "url": s.get("url")}
        for s in compact if s.get("url")
    ]
    return parsed


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Tata Dealer Finder", page_icon="🚗")

st.title("🚗 Tata Motors Dealer Finder (Tavily + Groq)")
st.write("Find the nearest Tata Motors dealership anywhere in India.")

user_query = st.text_input(
    "Enter your query",
    placeholder="Nearest Tata dealer in Wakad Pune"
)

if st.button("Find Dealer") and user_query:

    with st.spinner("Searching dealers..."):

        search_q = _build_search_query(user_query)
        results = tavily_search(search_q, max_results=6)
        response = groq_structured_answer(user_query, results)

    # --------------------------------------------------------
    # Display structured result
    # --------------------------------------------------------
    if response.get("mode") == "structured":

        if response.get("follow_up_question"):
            st.warning(response["follow_up_question"])

        dealers = response.get("dealers", [])

        if not dealers:
            st.info("No dealers found.")
        else:
            for dealer in dealers:
                with st.container():
                    st.subheader(dealer.get("name"))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("📍 Address:", dealer.get("address"))
                        st.write("📞 Phone:", dealer.get("phone"))
                        st.write("🕒 Hours:", dealer.get("hours"))

                    with col2:
                        st.write("🏙 City:", dealer.get("city"))
                        st.write("📌 Area:", dealer.get("area"))
                        st.write("🔎 Confidence:", dealer.get("confidence"))

                    if dealer.get("website"):
                        st.markdown(f"[Visit Website]({dealer.get('website')})")

                    st.markdown("---")

        # Optional debug
        with st.expander("Sources Used"):
            st.json(response.get("sources_used"))

    else:
        # Fallback display
        st.warning("Model returned unstructured response.")
        st.text(response.get("response"))