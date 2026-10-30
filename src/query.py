"""
Query understanding: turn "what is the outlook for South Asia?" into a metadata
filter (region / chapter) so search is restricted to the right part of the report.

Two layers:
1. Rules - explicit region names or abbreviations, "Chapter 3", "executive summary".
2. One Gemini call (free tier, cached on disk) for questions that name a country
   instead of a region. The report groups countries its own way (Pakistan sits in
   the Middle East, North Africa, Afghanistan, and Pakistan region), so the model
   is shown the region list. It may pick a chapter only when the question names one:
   a guessed chapter filter that is wrong hides the answer.
If the LLM is unavailable, the rules alone are used.
"""

import json
import os
import re

from src.structure import REGIONS

CACHE_PATH = "data/query_plan_cache.json"

CHAPTER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}


def rule_plan(question):
    q = question.lower()
    plan = {"region": None, "chapter": None, "tables": False, "source": "rules"}

    m = re.search(r"\b(?:chapter|ch\.?)\s*([1-4]|one|two|three|four)\b", q)
    if m:
        n = m.group(1)
        plan["chapter"] = f"ch{CHAPTER_WORDS.get(n, n)}"
    elif "executive summary" in q:
        plan["chapter"] = "exec"
    elif re.search(r"statistical (annex|appendix)", q):
        plan["chapter"] = "annex"

    hits = []
    for region, aliases in REGIONS.items():
        for a in aliases:
            if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", q):
                hits.append((len(a), region))
    if hits:
        # Longest alias wins ("middle east and north africa" over "mna")
        plan["region"] = max(hits)[1]
    plan["tables"] = bool(re.search(r"\b(table|annex|appendix|statistical)\b", q))
    return plan


def _load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=1)


LLM_PROMPT = """You route a question about the World Bank "Global Economic Prospects, June 2026" report.
Chapter 2 has one section per region. The report's regions, exactly:
{regions}
Note the report's own grouping: for example Pakistan and Afghanistan belong to
"Middle East, North Africa, Afghanistan, and Pakistan", not to South Asia.

Reply with JSON only: {{"region": <one region string from the list, or null>, "chapter": <"exec"|"ch1"|"ch2"|"ch3"|"ch4"|"annex"|null>}}
- region: set it only if the question asks about the economy, growth, or outlook of one region or of a country inside one region. If a country is only mentioned as an example inside a question about another topic (a box, a technology, a policy, a study), use null.
- chapter: set it ONLY if the question explicitly names a chapter or the executive summary. Never infer a chapter from the topic.

Question: {question}"""


def llm_plan(question, generate_json):
    prompt = LLM_PROMPT.format(regions="\n".join(f"- {r}" for r in REGIONS), question=question)
    raw = generate_json(prompt)
    data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    region = data.get("region")
    chapter = data.get("chapter")
    return {
        "region": region if region in REGIONS else None,
        "chapter": chapter if chapter in ("exec", "ch1", "ch2", "ch3", "ch4", "annex") else None,
    }


# A country named inside a question about something else (a box, a technology, a study)
# must not route to that country's region. Country-based routing therefore applies only
# when the question is about a regional economy.
ECONOMY_TERMS = re.compile(
    r"\b(growth|grow|outlook|forecast\w*|project\w*|economy|economies|economic|inflation|gdp|recovery|"
    r"recession|slowdown|prospects?|risks?|debt|fiscal|exports?|imports?|remittances)\b", re.I)
BOX_TERMS = re.compile(r"\bbox\b", re.I)


def make_plan(question, generate_json=None):
    plan = rule_plan(question)
    if plan["region"] or plan["chapter"] or generate_json is None:
        return plan
    cache = _load_cache()
    if question in cache:
        llm = cache[question]
    else:
        try:
            llm = llm_plan(question, generate_json)
        except Exception as e:  # network, quota, bad JSON: rules-only
            plan["note"] = f"llm routing unavailable ({type(e).__name__})"
            return plan
        cache[question] = llm
        _save_cache(cache)
    region = llm["region"]
    if region and (BOX_TERMS.search(question) or not ECONOMY_TERMS.search(question)):
        region = None
    plan.update(region=region, chapter=llm["chapter"], source="llm")
    return plan


if __name__ == "__main__":
    for q in ["What is the growth outlook for South Asia in this report?",
              "What does Chapter 3 say about sovereign debt in developing economies?",
              "According to the executive summary, what are the main risks to the global outlook?",
              "What is the outlook for Pakistan?"]:
        print(q, "->", rule_plan(q))
