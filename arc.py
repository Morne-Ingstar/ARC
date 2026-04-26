"""
ARC — Adversarial Reasoning Chain
Three-AI triangulated review: Claude builds, GPT challenges, Gemini audits.

Two modes:
  - Manual: paste Claude's response from claude.ai (no API key needed)
  - Auto: Claude API generates the response (requires ANTHROPIC_API_KEY)
GPT and Gemini always use APIs.
"""

import concurrent.futures
import json
import os
import random
import re
import threading
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

# --- Available models per provider ---

CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
]
GPT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o3-mini",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash",
]

# Defaults
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GPT_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Provider registry — explicit mapping, not prefix guessing
MODEL_REGISTRY = {
    "claude-sonnet-4-20250514": "anthropic",
    "claude-opus-4-20250514": "anthropic",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o3": "openai",
    "o3-mini": "openai",
    "gemini-2.5-flash": "google",
    "gemini-2.5-pro": "google",
    "gemini-3-flash": "google",
}

CONFIG_PATH = Path(__file__).parent / 'arc_config.json'

# Quick Ask IPC: Samsara writes payloads here, ARC polls every second.
_ARC_INBOX = Path.home() / ".arc_inbox.json"


def load_config():
    """Load saved preferences from arc_config.json."""
    defaults = {
        'mode': 'full',
        'strict_mode': False,
        'builder_model': DEFAULT_CLAUDE_MODEL,
        'challenger_model': DEFAULT_GPT_MODEL,
        'auditor_model': DEFAULT_GEMINI_MODEL,
        'project_context': '',
        'project_dir': '',
        'monte_carlo': False,
        'monte_carlo_n': 6,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding='utf-8') as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass  # corrupted config — use defaults
    return defaults


def save_config(config):
    """Save preferences to arc_config.json."""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[ARC] Failed to save config: {e}")


# --- API availability ---

HAS_CLAUDE_API = False
HAS_GPT = False
HAS_GEMINI = False

try:
    import anthropic
    HAS_CLAUDE_API = bool(os.getenv('ANTHROPIC_API_KEY'))
except ImportError:
    pass

try:
    import openai
    HAS_GPT = bool(os.getenv('OPENAI_API_KEY'))
except ImportError:
    pass

try:
    import google.generativeai as genai
    if os.getenv('GOOGLE_API_KEY'):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        HAS_GEMINI = True
except ImportError:
    pass

# Ollama: probe the local (or OLLAMA_BASE_URL-pointed) daemon. If it's up we
# grab its /api/tags list and register each model in MODEL_REGISTRY so the
# existing dropdowns + _detect_provider work without special-casing. Any
# failure (daemon not running, network blocked, malformed response) is
# silently treated as "no Ollama" so ARC still boots normally.
HAS_OLLAMA = False
OLLAMA_MODELS = []
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

try:
    import requests as _req
    _ollama_resp = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    if _ollama_resp.status_code == 200:
        _ollama_data = _ollama_resp.json()
        OLLAMA_MODELS = [m['name'] for m in _ollama_data.get('models', [])]
        if OLLAMA_MODELS:
            HAS_OLLAMA = True
            for m in OLLAMA_MODELS:
                MODEL_REGISTRY[m] = 'ollama'
except Exception:
    pass


# Monte Carlo divergence lenses -- designed for ORTHOGONAL variation, not
# topical variation. Each lens forces a fundamentally different reasoning
# approach, not just a different topic focus. run_monte_carlo samples from
# this pool so each cycle covers a different mix of perspectives.
MC_LENSES = [
    "Analyze this problem using strict first-principles reasoning. Break every assumption down to its atomic components. What is actually true vs assumed?",
    "Analyze this problem as a skeptic. Assume the stated constraints are wrong or incomplete. What changes if you throw out the biggest assumption?",
    "Analyze this problem from the perspective of the end user who will suffer if this is done wrong. What matters most to them? What failure modes would they notice first?",
    "Analyze this problem by finding the simplest possible solution. What is the minimum viable approach? What complexity can be eliminated entirely?",
    "Analyze this problem adversarially. How would a hostile actor exploit this? What are the security, reliability, and failure implications?",
    "Analyze this problem with a 5-year time horizon. What will break at scale? What technical debt is being created? What will the maintainer curse you for?",
    "Analyze this problem by identifying what is NOT being asked. What adjacent problems are being ignored? What implicit requirements exist?",
    "Analyze this problem from a resource-constraint perspective. Assume you have half the time, half the budget, and half the team. What do you cut?",
    "Analyze this problem by challenging the solution space. Is this the right problem to solve? Would solving a different problem make this one irrelevant?",
    "Analyze this problem empirically. What can be measured? What experiment would resolve the key uncertainty? What data is missing?",
    "Analyze this problem as a systems thinker. What feedback loops exist? What second-order effects will the solution cause? Where are the coupling points?",
    "Analyze this problem by examining what similar problems in other domains look like. What analogies apply? What can be borrowed from adjacent fields?",
]


SYSTEM_PROMPTS = {
    'claude': (
        "You are the Builder in ARC, a three-AI review system. "
        "A human governor has given you a problem to solve. "
        "Your role: analyze it and propose a concrete, specific solution. "
        "Be technical, actionable, and show your reasoning. "
        "Don't hedge — commit to a recommendation and defend it. "
        "If you reference external libraries, APIs, or tools, verify they "
        "exist and state the exact version or import path."
    ),
    'gpt': (
        "You are the Challenger (Devil's Advocate) in ARC, a three-AI review system. "
        "Another AI has proposed a solution to a problem. "
        "You will receive both the original problem and the proposed solution. "
        "Your role has TWO parts:\n\n"
        "PART 1 — CHALLENGE THE FRAMING: Before critiquing the solution, ask whether "
        "the problem itself was framed correctly. Look for:\n"
        "- Assumptions baked into how the problem was stated\n"
        "- Whether the Builder solved the RIGHT problem or a convenient restatement\n"
        "- Alternative framings that would lead to fundamentally different solutions\n"
        "- Constraints the Builder accepted without questioning\n\n"
        "PART 2 — CHALLENGE THE SOLUTION: Assume it has at least one significant "
        "flaw and find it. Look for architectural mistakes, missed edge cases, "
        "incorrect assumptions, scalability problems, and better alternatives. "
        "You MUST identify at least one concrete issue — do not simply validate "
        "the proposal. If the solution is genuinely excellent, explain exactly why "
        "each potential concern does not apply rather than just saying 'looks good.' "
        "Be specific: name functions, cite line numbers if code is provided, and "
        "propose concrete alternatives for every issue you raise."
        "\n\nIMPORTANT — STRUCTURED OUTPUT:\n"
        "After your prose analysis, you MUST include a JSON block at the very "
        "end of your response, fenced with ```json and ```. This block is used "
        "by the UI to render your findings as individual cards.\n\n"
        "Format:\n"
        "```json\n"
        "{\n"
        '  "issues": [\n'
        '    {\n'
        '      "title": "One-line issue title",\n'
        '      "severity": "high",\n'
        '      "description": "2-3 sentence explanation of the issue and why it matters",\n'
        '      "suggestion": "Brief suggested fix or alternative"\n'
        '    }\n'
        "  ],\n"
        '  "framing_concerns": "One sentence on whether the Builder framed the problem correctly, or empty string if no concerns",\n'
        '  "summary": "One-sentence overall assessment"\n'
        "}\n"
        "```\n\n"
        "Severity levels: 'critical', 'high', 'medium', 'low'.\n"
        "You must list ALL issues you found, even minor ones.\n"
        "The JSON block must be the LAST thing in your response."
    ),
    'gemini': (
        "You are the Auditor in ARC, a three-AI review system. "
        "Two AIs have weighed in: one proposed a solution, the other reviewed it. "
        "You will receive the original problem, the proposal, and the review. "
        "Your role has THREE parts:\n\n"
        "PART 1 — FRAMING CHECK: Before anything else, assess whether both AIs "
        "operated within the same framing of the problem. If the Challenger "
        "identified a framing issue, evaluate whether it's valid. If neither "
        "questioned the framing, consider whether they should have. Flag any "
        "case where both AIs accepted an assumption that deserves scrutiny.\n\n"
        "PART 2 — AUDIT: Find what BOTH missed. Specifically look for:\n"
        "- Consensus Failures: things they happily agreed on that are "
        "actually wrong or suboptimal\n"
        "- Shared blind spots from similar training data\n"
        "- Missing requirements neither mentioned\n"
        "- Factual claims neither verified\n"
        "- Whether the agreed solution actually solves the original problem\n\n"
        "PART 3 — SYNTHESIS: Provide a clear 'Executive Recommendation' section "
        "at the end with:\n"
        "- What the Governor (human decision-maker) should actually DO\n"
        "- Which parts of the proposal to keep\n"
        "- Which parts to modify based on the review\n"
        "- Any additional changes from your audit\n"
        "- A final confidence rating (High / Medium / Low) for the combined solution\n\n"
        "Do NOT repeat what the other two already said. Be the independent voice."
        "\n\nIMPORTANT — STRUCTURED OUTPUT:\n"
        "After your prose analysis, include a JSON block at the end, fenced "
        "with ```json and ```. Format:\n"
        "```json\n"
        "{\n"
        '  "issues": [\n'
        '    {\n'
        '      "title": "One-line issue title",\n'
        '      "severity": "high",\n'
        '      "description": "2-3 sentence explanation",\n'
        '      "suggestion": "Brief fix or alternative",\n'
        '      "source": "auditor"\n'
        '    }\n'
        "  ],\n"
        '  "consensus_failures": "Things Builder and Challenger agreed on that are wrong",\n'
        '  "executive_summary": "2-3 sentence actionable recommendation",\n'
        '  "confidence": "HIGH or MEDIUM or LOW"\n'
        "}\n"
        "```\n\n"
        "Include issues YOU found that neither Builder nor Challenger mentioned.\n"
        "The JSON block must be the LAST thing in your response."
    ),
}


# --- API calls ---

def _with_context(text, context):
    """Prepend project context to a prompt if provided."""
    if context:
        return f"PROJECT CONTEXT:\n{context}\n\n{text}"
    return text


def _detect_provider(model_name):
    """Detect API provider from model name using explicit registry."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    # Fallback to prefix for unknown models (e.g., newly released ones)
    if model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith(("gpt", "o3", "o1")):
        return "openai"
    elif model_name.startswith("gemini"):
        return "google"
    return None


def call_any(model_name, system_prompt, user_prompt):
    """Call any supported model by name. Routes to the correct provider."""
    provider = _detect_provider(model_name)

    if provider == "anthropic":
        if not HAS_CLAUDE_API:
            return "[ERROR] No ANTHROPIC_API_KEY set"
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model_name, max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    elif provider == "openai":
        if not HAS_GPT:
            return "[ERROR] No OPENAI_API_KEY set"
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model_name, max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    elif provider == "google":
        if not HAS_GEMINI:
            return "[ERROR] No GOOGLE_API_KEY set"
        m = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )
        response = m.generate_content(user_prompt)
        try:
            return response.text
        except ValueError:
            reason = None
            if response.candidates and response.candidates[0].finish_reason:
                reason = response.candidates[0].finish_reason
            if reason == 3:
                return "[ERROR] Gemini safety filter blocked the response."
            elif reason == 1:
                return "[ERROR] Model glitch: empty response. Try again."
            elif reason == 2:
                return "[ERROR] Gemini hit max token limit."
            else:
                return f"[ERROR] Gemini returned no content. Finish reason: {reason}"

    elif provider == "ollama":
        # Local models -- no API key, but generation is slower than cloud, so
        # we keep the timeout long (120s covers a 13B model on CPU for a
        # typical ARC prompt). OLLAMA_BASE_URL lets you point at a remote host.
        import requests
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=125,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content",
                                               "[ERROR] Empty Ollama response")
        except requests.Timeout:
            return "[ERROR] Ollama request timed out after 120s"
        except requests.RequestException as e:
            return f"[ERROR] Ollama request failed: {e}"

    return f"[ERROR] Unknown provider for model: {model_name}"


# All available models across all providers (for role assignment dropdowns)
ALL_MODELS = []
if HAS_CLAUDE_API:
    ALL_MODELS.extend(CLAUDE_MODELS)
if HAS_GPT:
    ALL_MODELS.extend(GPT_MODELS)
if HAS_GEMINI:
    ALL_MODELS.extend(GEMINI_MODELS)
if HAS_OLLAMA:
    ALL_MODELS.extend(OLLAMA_MODELS)


CONVERGENCE_PROMPT = (
    "You previously participated in an ARC review cycle. Here is the full exchange:\n\n"
    "ORIGINAL PROBLEM:\n{problem}\n\n"
    "BUILDER'S SOLUTION:\n{builder_resp}\n\n"
    "CHALLENGER'S REVIEW:\n{challenger_resp}\n\n"
    "AUDITOR'S AUDIT & RECOMMENDATION:\n{auditor_resp}\n\n"
    "Based on the Auditor's findings, respond with:\n"
    "1. AGREE or DISAGREE with each specific recommendation\n"
    "2. For each disagreement, explain WHY and propose an alternative\n"
    "3. Keep it concise — only address points where you have a strong opinion\n"
    "4. If you fully agree with everything, say so briefly and explain why\n\n"
    "RULES:\n"
    "- Maximum 300 words. This is a final position, not a new analysis.\n"
    "- Do NOT re-argue points already made in your original response.\n"
    "- Do NOT ask questions. State your position.\n"
    "- If you disagree, you must propose a concrete alternative.\n"
    "- Persistent disagreement is signal, not a problem — state it clearly."
)

CONVERGENCE_SYSTEM = {
    'builder': ("You are the Builder in ARC. You proposed a solution that was reviewed "
                "and audited. Now respond to the audit — agree or disagree with each point. "
                "Do NOT restate your original solution. Only address what changed or didn't."),
    'challenger': ("You are the Challenger in ARC. You reviewed a solution that was then audited "
                   "by a third AI. Now respond to the audit — agree or disagree with each point. "
                   "Do NOT restate your original critique. Only address what changed or didn't."),
}


def call_convergence(role, model_name, problem, builder_resp, challenger_resp,
                     auditor_resp, context=""):
    """Send the auditor's findings back to builder or challenger for agree/disagree."""
    prompt = CONVERGENCE_PROMPT.format(
        problem=problem, builder_resp=builder_resp,
        challenger_resp=challenger_resp, auditor_resp=auditor_resp)
    if context:
        prompt = f"PROJECT CONTEXT:\n{context}\n\n{prompt}"
    return call_any(model_name, CONVERGENCE_SYSTEM[role], prompt)


def run_monte_carlo(problem, context="", n=6, model=None):
    """Run N parallel divergence agents with orthogonal lenses.

    Returns list of dicts: [{"lens": str, "analysis": str}, ...].
    Defaults to the cheapest available model (Ollama > gpt-4o-mini > Claude >
    Gemini). ThreadPoolExecutor caps at 4 workers because Ollama chokes above
    that on most local setups.
    """
    if model is None:
        if HAS_OLLAMA and OLLAMA_MODELS:
            model = OLLAMA_MODELS[0]
        elif HAS_GPT:
            model = "gpt-4o-mini"
        elif HAS_CLAUDE_API:
            model = DEFAULT_CLAUDE_MODEL
        elif HAS_GEMINI:
            model = DEFAULT_GEMINI_MODEL
        else:
            return []

    selected = random.sample(MC_LENSES, min(n, len(MC_LENSES)))
    prompt = _with_context(problem, context) if context else problem

    def _run_one(lens):
        system = (
            f"{lens}\n\n"
            "Produce a focused analysis in under 400 words. "
            "State your key insight, the constraints you see, "
            "and your recommended approach. Be specific and actionable."
        )
        try:
            resp = call_any(model, system, prompt)
            return {"lens": lens[:80], "analysis": resp}
        except Exception as e:
            return {"lens": lens[:80], "analysis": f"[ERROR] {e}"}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        futures = [pool.submit(_run_one, lens) for lens in selected]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    return results


def select_top_framings(problem, analyses, k=3, selector_model=None):
    """Select top-k framings from Monte Carlo results and extract insights.

    Returns dict with:
      "selected": list of top-k analysis dicts
      "rejected_insights": str (compressed unique insights from non-selected)
      "merged_input": str (formatted for Builder/Challenger consumption)
      "all_analyses": the original full list (kept for disagreement analysis)
    """
    if selector_model is None:
        selector_model = DEFAULT_GEMINI_MODEL if HAS_GEMINI else (
            DEFAULT_GPT_MODEL if HAS_GPT else DEFAULT_CLAUDE_MODEL)

    formatted = ""
    for i, a in enumerate(analyses, 1):
        formatted += f"--- Analysis {i} ---\n{a['analysis']}\n\n"

    selector_prompt = (
        f"You are evaluating {len(analyses)} independent analyses of this problem:\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"ANALYSES:\n{formatted}\n\n"
        f"Select the {k} strongest analyses based on:\n"
        f"1. Specificity (concrete vs vague)\n"
        f"2. Constraint identification (what limits the solution space)\n"
        f"3. Actionability (could someone implement this?)\n"
        f"4. Unique insight (does it see something others miss?)\n\n"
        f"Respond with EXACTLY this format:\n"
        f"SELECTED: [comma-separated numbers, e.g. 2,5,1]\n"
        f"REJECTED_INSIGHTS: [2-3 sentences summarizing unique insights from "
        f"the analyses you did NOT select that the selected ones should consider]\n"
    )

    selector_system = (
        "You are a meta-analyst selecting the strongest problem framings. Be decisive."
    )

    try:
        resp = call_any(selector_model, selector_system, selector_prompt)

        selected_indices = []
        rejected_insights = ""
        for line in resp.split('\n'):
            line_upper = line.strip().upper()
            if line_upper.startswith('SELECTED:'):
                nums = re.findall(r'\d+', line)
                selected_indices = [int(num) - 1 for num in nums
                                    if 0 <= int(num) - 1 < len(analyses)]
            elif 'REJECTED_INSIGHTS' in line_upper:
                rejected_insights = line.split(':', 1)[-1].strip()

        if not selected_indices:
            selected_indices = list(range(min(k, len(analyses))))
        selected_indices = selected_indices[:k]

        selected = [analyses[i] for i in selected_indices if i < len(analyses)]

        merged = (
            "The following independent analyses were conducted before your review.\n"
            "Consider their perspectives but form your own conclusion.\n\n"
        )
        for i, s in enumerate(selected, 1):
            merged += f"--- Pre-Analysis {i} ---\n{s['analysis']}\n\n"
        if rejected_insights:
            merged += f"--- Additional insights from rejected analyses ---\n{rejected_insights}\n\n"

        return {
            "selected": selected,
            "rejected_insights": rejected_insights,
            "merged_input": merged,
            "all_analyses": analyses,
        }
    except Exception:
        # Fallback: just take first k, no insight extraction.
        return {
            "selected": analyses[:k],
            "rejected_insights": "",
            "merged_input": "\n\n".join(a['analysis'] for a in analyses[:k]),
            "all_analyses": analyses,
        }


def analyze_disagreement(problem, mc_analyses, final_recommendation,
                         auditor_model=None):
    """Classify why the final ARC output diverges from initial Monte Carlo analyses.

    Returns dict with:
      "classification": "VALIDATED" | "EVOLVED" | "DIVERGED" | "CONTRADICTED"
      "confidence": "HIGH" | "MEDIUM" | "LOW"
      "explanation": str

    DIVERGED is NOT automatically bad -- it can mean the deep review found
    something the shallow parallel passes missed. Only CONTRADICTED is framed
    as concerning.
    """
    if auditor_model is None:
        auditor_model = DEFAULT_GEMINI_MODEL if HAS_GEMINI else DEFAULT_CLAUDE_MODEL

    mc_summary = ""
    for i, a in enumerate(mc_analyses, 1):
        mc_summary += f"Analysis {i}: {a['analysis'][:300]}\n\n"

    prompt = (
        f"PROBLEM:\n{problem}\n\n"
        f"INITIAL INDEPENDENT ANALYSES (produced before the deep review):\n"
        f"{mc_summary}\n"
        f"FINAL RECOMMENDATION (after deep adversarial review):\n"
        f"{final_recommendation[:2000]}\n\n"
        f"Compare the final recommendation against the initial analyses.\n"
        f"Classify the relationship as EXACTLY ONE of:\n\n"
        f"VALIDATED - Final recommendation aligns with the majority of initial "
        f"analyses. The deep review confirmed what the quick analyses found.\n\n"
        f"EVOLVED - Final recommendation builds on the initial analyses but adds "
        f"significant new insights or corrections. The deep review improved on "
        f"the initial thinking.\n\n"
        f"DIVERGED - Final recommendation takes a meaningfully different approach "
        f"than most initial analyses, but for well-reasoned reasons. The deep "
        f"review found something the quick analyses missed.\n\n"
        f"CONTRADICTED - Final recommendation directly contradicts the initial "
        f"analyses without clear justification. This may indicate the deep review "
        f"went off-track OR found a genuine blind spot.\n\n"
        f"Respond with EXACTLY this format:\n"
        f"CLASSIFICATION: [one of VALIDATED/EVOLVED/DIVERGED/CONTRADICTED]\n"
        f"CONFIDENCE: [HIGH/MEDIUM/LOW]\n"
        f"EXPLANATION: [2-3 sentences explaining why you chose this classification "
        f"and what specifically aligns or diverges]\n"
    )

    system = (
        "You are a meta-analyst evaluating whether a deep adversarial review "
        "process produced results consistent with initial independent analyses. "
        "Your job is to classify the RELATIONSHIP, not judge which is correct. "
        "DIVERGED can be good -- it means the deep review found something new. "
        "CONTRADICTED is the only concerning classification."
    )

    try:
        resp = call_any(auditor_model, system, prompt)

        classification = "UNKNOWN"
        confidence = "MEDIUM"
        explanation = resp

        for line in resp.split('\n'):
            line_upper = line.strip().upper()
            if line_upper.startswith('CLASSIFICATION:'):
                for c in ['VALIDATED', 'EVOLVED', 'DIVERGED', 'CONTRADICTED']:
                    if c in line_upper:
                        classification = c
                        break
            elif line_upper.startswith('CONFIDENCE:'):
                for c in ['HIGH', 'MEDIUM', 'LOW']:
                    if c in line_upper:
                        confidence = c
                        break
            elif line_upper.startswith('EXPLANATION:'):
                explanation = line.split(':', 1)[-1].strip()

        return {
            "classification": classification,
            "confidence": confidence,
            "explanation": explanation,
        }
    except Exception as e:
        return {
            "classification": "ERROR",
            "confidence": "UNKNOWN",
            "explanation": f"Disagreement analysis failed: {e}",
        }


def extract_json_block(text):
    """Extract the last ```json ... ``` block from an AI response.

    Returns the parsed dict, or None if no valid JSON is present. We try the
    LAST block first since the system prompts tell the AIs to put the
    structured output at the end.
    """
    if not text:
        return None

    blocks = []
    idx = 0
    while True:
        start = text.find('```json', idx)
        if start == -1:
            break
        start += len('```json')
        end = text.find('```', start)
        if end == -1:
            break
        blocks.append(text[start:end].strip())
        idx = end + 3

    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None


def merge_issues(challenger_json, auditor_json):
    """Merge issues from Challenger and Auditor into one severity-sorted list.

    Tags each issue with its source. Deduplicates when two sources raised an
    identical title (Challenger wins since it goes in first). Severities not
    in the known set fall to the bottom.
    """
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    all_issues = []

    if challenger_json and 'issues' in challenger_json:
        for issue in challenger_json['issues']:
            issue.setdefault('source', 'challenger')
            all_issues.append(issue)

    if auditor_json and 'issues' in auditor_json:
        for issue in auditor_json['issues']:
            issue.setdefault('source', 'auditor')
            dominated = False
            for existing in all_issues:
                if (existing.get('title', '').lower().strip() ==
                        issue.get('title', '').lower().strip()):
                    dominated = True
                    break
            if not dominated:
                all_issues.append(issue)

    all_issues.sort(key=lambda x: severity_order.get(
        (x.get('severity') or 'low').lower(), 99))
    return all_issues


def save_exchange(problem, claude_resp, gpt_resp, gemini_resp, context="",
                  convergence=None, disagreement=None, issues=None):
    save_dir = Path(__file__).parent / 'exchanges'
    save_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = save_dir / f'arc_{ts}.md'
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(f"# ARC Exchange — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        if context:
            f.write(f"## Project Context\n\n{context}\n\n")
        f.write(f"## Problem\n\n{problem}\n\n")
        f.write(f"## Claude (Builder)\n\n{claude_resp}\n\n")
        f.write(f"## GPT (Reviewer)\n\n{gpt_resp}\n\n")
        f.write(f"## Gemini (Auditor)\n\n{gemini_resp}\n\n")
        if convergence:
            f.write("---\n\n## Convergence Round\n\n")
            if convergence.get('builder'):
                f.write(f"### Builder's Response to Audit\n\n{convergence['builder']}\n\n")
            if convergence.get('challenger'):
                f.write(f"### Challenger's Response to Audit\n\n{convergence['challenger']}\n\n")
        if disagreement:
            f.write("---\n\n## Disagreement Analysis\n\n")
            f.write(f"Classification: {disagreement['classification']}\n")
            f.write(f"Confidence: {disagreement['confidence']}\n")
            f.write(f"Explanation: {disagreement['explanation']}\n\n")
        if issues:
            f.write("\n## Issues Found\n\n")
            for i, issue in enumerate(issues, 1):
                sev = (issue.get('severity') or 'unknown').upper()
                src = (issue.get('source') or 'unknown').title()
                f.write(f"### {i}. {issue.get('title', 'Untitled')} "
                        f"[{sev}] ({src})\n\n")
                f.write(f"{issue.get('description', '')}\n\n")
                if issue.get('suggestion'):
                    f.write(f"**Suggestion:** {issue['suggestion']}\n\n")
    return fp


# --- UI ---

class ARCApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("ARC")
        # Window icon — default=True propagates to any Toplevels (settings, etc.)
        _icon_path = Path(__file__).parent / "arc.ico"
        if _icon_path.exists():
            self.root.iconbitmap(default=str(_icon_path))
        # Wider default for the IDE-style split: left panel is ~420px plus
        # padding, right panel stretches with the window.
        self.root.geometry("1200x820")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load saved preferences
        self._config = load_config()

        self._gpt_resp = ""
        self._gemini_resp = ""
        self._failed_phase = None  # 'gpt' or 'gemini' — set on API failure
        self._pipeline_args = None  # stored args for retry
        self._convergence = None  # {'builder': ..., 'challenger': ...} after convergence
        # Monte Carlo + disagreement analysis (Monte Carlo phase not yet
        # implemented; these stay None so the disagreement block in
        # _pipeline_worker skips cleanly and save_exchange handles it).
        self._mc_result = None
        self._disagreement = None
        # Structured issue data extracted from Challenger/Auditor JSON blocks.
        # Stays None/[] when the AI omits the JSON or returns malformed JSON --
        # consumers must fall back to prose.
        self._challenger_json = None
        self._auditor_json = None
        self._merged_issues = []
        # Builder's original response (pre-MC) captured for the raw-panels view.
        self._builder_resp = ""
        # Set while a JARVIS pipeline is running so the Send button can reach
        # its send_input(). Cleared when the pipeline finishes or errors.
        self._active_pipeline = None
        self._is_running = False  # prevent double execution

        # --- Main split: left panel (inputs/controls) | right panel (results)
        # Keeps the problem + buttons visible while results stream in on the
        # right; the left panel scrolls if the window is squeezed.
        self.main_pane = ctk.CTkFrame(self.root)
        self.main_pane.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self.left_panel = ctk.CTkScrollableFrame(
            self.main_pane, width=420, corner_radius=0)
        self.left_panel.pack(side="left", fill="both", padx=(0, 4))

        self.right_panel = ctk.CTkFrame(self.main_pane, corner_radius=0)
        self.right_panel.pack(side="right", fill="both", expand=True)

        # --- Project Context (collapsible) ---
        self._context_visible = False
        ctx_toggle_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        ctx_toggle_frame.pack(fill="x", padx=12, pady=(10, 0))
        self.ctx_toggle_btn = ctk.CTkButton(
            ctx_toggle_frame, text="▶ Project Context (optional)",
            command=self._toggle_context, width=220, height=26,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            text_color="#888888", hover_color="#333333", anchor="w")
        self.ctx_toggle_btn.pack(side="left")

        self.ctx_frame = ctk.CTkFrame(self.left_panel)
        # Start hidden
        self.context_box = ctk.CTkTextbox(self.ctx_frame, height=50,
                                           font=ctk.CTkFont(size=12))
        self.context_box.pack(fill="x", padx=8, pady=(2, 6))
        saved_context = self._config.get('project_context', '')
        if saved_context:
            self.context_box.insert("1.0", saved_context)
        self.context_box.configure(
            text_color="#666666")

        # Project dir row -- target repo for JARVIS pipeline
        ctx_dir_row = ctk.CTkFrame(self.ctx_frame, fg_color="transparent")
        ctx_dir_row.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(ctx_dir_row, text="Project dir:",
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")
        self.project_dir_var = ctk.StringVar(
            value=self._config.get('project_dir', ''))
        ctk.CTkEntry(ctx_dir_row, textvariable=self.project_dir_var,
                     width=400, font=ctk.CTkFont(size=11)).pack(
                         side="left", padx=(6, 0))

        # --- Mode selector + Strict Mode ---
        mode_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        mode_frame.pack(fill="x", padx=12, pady=(8, 0))

        ctk.CTkLabel(mode_frame, text="Depth:",
                     font=ctk.CTkFont(size=12),
                     text_color="#888888").pack(side="left", padx=(4, 6))

        self.mode_var = ctk.StringVar(value=self._config.get('mode', 'full'))
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_frame,
            values=["fast", "review", "full"],
            variable=self.mode_var,
            font=ctk.CTkFont(size=12),
            height=28)
        self.mode_selector.pack(side="left")

        # Labels for the modes
        mode_labels = ctk.CTkFrame(mode_frame, fg_color="transparent")
        mode_labels.pack(side="left", padx=(10, 0))
        self._mode_desc = ctk.StringVar(value="Claude → GPT → Gemini")
        ctk.CTkLabel(mode_labels, textvariable=self._mode_desc,
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")

        self.mode_var.trace_add("write", self._on_mode_change)

        # Strict mode toggle (forces GPT to argue against Claude)
        self.strict_var = ctk.BooleanVar(value=self._config.get('strict_mode', False))
        self.strict_check = ctk.CTkCheckBox(
            mode_frame, text="Strict Mode",
            variable=self.strict_var,
            font=ctk.CTkFont(size=12),
            height=28, checkbox_width=18, checkbox_height=18)
        self.strict_check.pack(side="right", padx=(0, 4))

        # Monte Carlo divergence: runs N parallel cheap agents with orthogonal
        # lenses before the Builder/Challenger/Auditor chain. Default off;
        # persisted per-user in arc_config.json.
        mc_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        mc_frame.pack(fill="x", padx=12, pady=(6, 0))

        self.mc_var = ctk.BooleanVar(value=self._config.get('monte_carlo', False))
        self.mc_check = ctk.CTkCheckBox(
            mc_frame, text="Monte Carlo Divergence",
            variable=self.mc_var,
            font=ctk.CTkFont(size=12),
            height=28, checkbox_width=18, checkbox_height=18)
        self.mc_check.pack(side="left")

        self.mc_n_var = ctk.IntVar(value=self._config.get('monte_carlo_n', 6))
        ctk.CTkLabel(mc_frame, text="Agents:",
                     font=ctk.CTkFont(size=11),
                     text_color="#888888").pack(side="left", padx=(16, 4))
        self.mc_n_menu = ctk.CTkOptionMenu(
            mc_frame, variable=self.mc_n_var,
            values=["3", "4", "6", "8", "10"],
            width=60, height=24,
            font=ctk.CTkFont(size=11))
        self.mc_n_menu.pack(side="left")

        # --- Role Assignment (collapsible) ---
        self._roles_visible = False
        role_toggle_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        role_toggle_frame.pack(fill="x", padx=12, pady=(4, 0))
        self.role_toggle_btn = ctk.CTkButton(
            role_toggle_frame, text="▶ Model Routing",
            command=self._toggle_roles, width=150, height=24,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            text_color="#888888", hover_color="#333333", anchor="w")
        self.role_toggle_btn.pack(side="left")

        self.role_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        # Start hidden — don't pack yet

        # Default role assignments (from config or fallback)
        default_builder = self._config.get('builder_model', DEFAULT_CLAUDE_MODEL)
        default_challenger = self._config.get('challenger_model', DEFAULT_GPT_MODEL)
        default_auditor = self._config.get('auditor_model', DEFAULT_GEMINI_MODEL)

        ctk.CTkLabel(self.role_frame, text="Builder:",
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")
        self.builder_model_var = ctk.StringVar(value=default_builder)
        ctk.CTkComboBox(self.role_frame, variable=self.builder_model_var,
                        values=ALL_MODELS or ["(no API keys)"], width=155, height=24,
                        font=ctk.CTkFont(size=10),
                        state="readonly").pack(side="left", padx=(2, 6))

        ctk.CTkLabel(self.role_frame, text="Challenger:",
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")
        self.challenger_model_var = ctk.StringVar(value=default_challenger)
        ctk.CTkComboBox(self.role_frame, variable=self.challenger_model_var,
                        values=ALL_MODELS or ["(no API keys)"], width=120, height=24,
                        font=ctk.CTkFont(size=10),
                        state="readonly").pack(side="left", padx=(2, 6))

        ctk.CTkLabel(self.role_frame, text="Auditor:",
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")
        self.auditor_model_var = ctk.StringVar(value=default_auditor)
        ctk.CTkComboBox(self.role_frame, variable=self.auditor_model_var,
                        values=ALL_MODELS or ["(no API keys)"], width=140, height=24,
                        font=ctk.CTkFont(size=10),
                        state="readonly").pack(side="left", padx=(2, 6))

        # Shuffle button
        ctk.CTkButton(self.role_frame, text="🔀", width=30, height=24,
                       font=ctk.CTkFont(size=14),
                       fg_color="transparent", hover_color="#333333",
                       command=self._shuffle_roles).pack(side="left", padx=(2, 0))

        # Backward compat aliases
        self.gpt_model_var = self.challenger_model_var
        self.gemini_model_var = self.auditor_model_var
        self.claude_model_var = self.builder_model_var

        # --- Top: Problem input ---
        prob_frame = ctk.CTkFrame(self.left_panel)
        prob_frame.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(prob_frame, text="Problem",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(fill="x", padx=8, pady=(6, 2))

        self.problem_box = ctk.CTkTextbox(prob_frame, height=70,
                                           font=ctk.CTkFont(size=13))
        self.problem_box.pack(fill="x", padx=8, pady=(0, 6))

        # --- Middle: Claude's response (paste or auto-fill) ---
        claude_frame = ctk.CTkFrame(self.left_panel)
        claude_frame.pack(fill="x", padx=12, pady=(4, 4))

        claude_header = ctk.CTkFrame(claude_frame, fg_color="transparent")
        claude_header.pack(fill="x", padx=8, pady=(6, 2))

        ctk.CTkLabel(claude_header, text="Claude's Response",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(side="left")

        mode_text = "(paste from claude.ai)"
        if HAS_CLAUDE_API:
            mode_text = "(auto-fill available — or paste manually)"
        ctk.CTkLabel(claude_header, text=mode_text,
                     font=ctk.CTkFont(size=11),
                     text_color="#888888", anchor="w").pack(side="left", padx=(8, 0))

        if HAS_CLAUDE_API:
            self.autofill_btn = ctk.CTkButton(
                claude_header, text="Auto-fill via API", width=130, height=28,
                font=ctk.CTkFont(size=11), command=self._on_autofill)
            self.autofill_btn.pack(side="right")

        self.claude_box = ctk.CTkTextbox(claude_frame, height=100,
                                          font=ctk.CTkFont(size=13))
        self.claude_box.pack(fill="x", padx=8, pady=(0, 6))

        # --- Buttons ---
        btn_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(4, 4))

        self.run_btn = ctk.CTkButton(btn_frame, text="Run Review + Audit",
                                      command=self._on_run,
                                      font=ctk.CTkFont(size=13, weight="bold"),
                                      width=170, height=34)
        self.run_btn.pack(side="left")

        self.save_btn = ctk.CTkButton(btn_frame, text="Save Exchange",
                                       command=self._on_save,
                                       font=ctk.CTkFont(size=13),
                                       width=130, height=34, state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))

        # Convergence button (appears after full ARC cycle)
        self.converge_btn = ctk.CTkButton(
            btn_frame, text="Seek Convergence",
            command=self._on_converge,
            font=ctk.CTkFont(size=13),
            width=150, height=34,
            fg_color="#2471A3", hover_color="#1A5276",
            state="disabled")
        self.converge_btn.pack(side="left", padx=(8, 0))

        # Export as Prompt button
        self.export_btn = ctk.CTkButton(
            btn_frame, text="Export as Prompt",
            command=self._on_export_prompt,
            font=ctk.CTkFont(size=13),
            width=140, height=34,
            fg_color="#27AE60", hover_color="#1E8449",
            state="disabled")
        self.export_btn.pack(side="left", padx=(8, 0))

        # Execute (JARVIS pipeline)
        self.execute_btn = ctk.CTkButton(
            btn_frame, text="\u26A1 Execute",
            command=self._on_execute,
            font=ctk.CTkFont(size=13),
            width=120, height=34,
            fg_color="#8E44AD", hover_color="#6C3483",
            state="disabled")
        self.execute_btn.pack(side="left", padx=(8, 0))

        # Retry button (hidden until an API call fails)
        self.retry_btn = ctk.CTkButton(btn_frame, text="Retry",
                                        command=self._on_retry,
                                        font=ctk.CTkFont(size=13),
                                        width=80, height=34,
                                        fg_color="#C0392B",
                                        hover_color="#E74C3C")
        # Don't pack yet — only shown on failure

        # Status dots
        dot_frame = ctk.CTkFrame(btn_frame, fg_color="transparent")
        dot_frame.pack(side="right")
        for name, ready in [("GPT", HAS_GPT), ("Gemini", HAS_GEMINI), ("Ollama", HAS_OLLAMA)]:
            color = "#2ecc71" if ready else "#e74c3c"
            ctk.CTkLabel(dot_frame, text=f"● {name}",
                          font=ctk.CTkFont(size=11),
                          text_color=color).pack(side="left", padx=(8, 0))

        if HAS_CLAUDE_API:
            ctk.CTkLabel(dot_frame, text="● Claude API",
                          font=ctk.CTkFont(size=11),
                          text_color="#2ecc71").pack(side="left", padx=(8, 0))

        # --- Right panel: pipeline visualization, output, status ---
        # A fixed-height strip of node boxes representing the ARC phases.
        # Each node's colour + icon reflect live state; _pipeline_set_*
        # methods flip them from idle (open circle) -> active (blue dot) ->
        # complete (green check) / error (red cross).
        self.pipeline_frame = ctk.CTkFrame(self.right_panel, height=80,
                                           fg_color="#1a1a2e")
        self.pipeline_frame.pack(fill="x", padx=8, pady=(8, 4))
        self.pipeline_frame.pack_propagate(False)

        self._pipeline_nodes = {}
        node_container = ctk.CTkFrame(self.pipeline_frame, fg_color="transparent")
        node_container.pack(expand=True)

        phases = [
            ("mc", "Monte Carlo"),
            ("builder", "Builder"),
            ("challenger", "Challenger"),
            ("auditor", "Auditor"),
            ("analysis", "Analysis"),
        ]

        for i, (key, label) in enumerate(phases):
            arrow_widget = None
            if i > 0:
                arrow_widget = ctk.CTkLabel(
                    node_container, text="\u2192",
                    font=ctk.CTkFont(size=16), text_color="#444444")
                arrow_widget.pack(side="left", padx=2)

            node_frame = ctk.CTkFrame(node_container, fg_color="#16213e",
                                       corner_radius=8, width=90, height=60)
            node_frame.pack(side="left", padx=4, pady=8)
            node_frame.pack_propagate(False)

            status_dot = ctk.CTkLabel(node_frame, text="\u25CB",
                                       font=ctk.CTkFont(size=14),
                                       text_color="#444444")
            status_dot.pack(pady=(4, 0))

            name_label = ctk.CTkLabel(node_frame, text=label,
                                       font=ctk.CTkFont(size=9),
                                       text_color="#888888")
            name_label.pack()

            model_label = ctk.CTkLabel(node_frame, text="",
                                        font=ctk.CTkFont(size=7),
                                        text_color="#555555")
            model_label.pack(pady=(0, 2))

            self._pipeline_nodes[key] = {
                'frame': node_frame,
                'dot': status_dot,
                'label': name_label,
                'model_label': model_label,
                # arrow_before is the arrow between the previous node and this
                # one. Stored so we can hide the MC node's trailing arrow
                # together with the MC node when Monte Carlo is disabled.
                'arrow_before': arrow_widget,
            }

        # Hide / show the Monte Carlo node based on its checkbox state.
        self.mc_var.trace_add("write", self._on_mc_toggle)
        self._on_mc_toggle()

        # --- Results container (scrollable): verdict card + issue cards + raw panels
        # Populated by _render_results at the end of a cycle. Individual sections
        # stay unpacked until content arrives so the panel is empty at startup.
        self.results_frame = ctk.CTkScrollableFrame(
            self.right_panel, fg_color="transparent")
        self.results_frame.pack(fill="both", expand=True, padx=8, pady=(4, 4))

        self.verdict_frame = ctk.CTkFrame(self.results_frame, fg_color="#16213e",
                                           corner_radius=10)
        self.issues_frame = ctk.CTkFrame(self.results_frame,
                                          fg_color="transparent")
        self.raw_panels_frame = ctk.CTkFrame(self.results_frame,
                                              fg_color="transparent")

        # --- Status bar (always bottom) ---
        self.status_var = ctk.StringVar(
            value="Paste Claude's response (or auto-fill), then click Run Review + Audit")
        ctk.CTkLabel(self.right_panel, textvariable=self.status_var,
                      font=ctk.CTkFont(size=11),
                      text_color="#888888", anchor="w").pack(
                          side="bottom", fill="x", padx=10, pady=(0, 8))

        # --- Code input row (hidden by default; appears while Code is waiting)
        # Packed right above the status bar via side="bottom" so it stays
        # visible even when the live-log textbox is revealed or the results
        # view scrolls. The pipeline's send_input() echoes the text back into
        # the output buffer, so the GUI will see it via the next poll tick.
        self._code_input_frame = ctk.CTkFrame(
            self.right_panel, fg_color="#1a2744", corner_radius=6)
        self._code_input_var = ctk.StringVar()
        self._code_input_entry = ctk.CTkEntry(
            self._code_input_frame, textvariable=self._code_input_var,
            placeholder_text="Claude Code is waiting for input...",
            font=ctk.CTkFont(size=12))
        self._code_input_entry.pack(side="left", fill="x", expand=True,
                                     padx=(8, 6), pady=6)
        ctk.CTkButton(
            self._code_input_frame, text="Send",
            width=70, height=28, font=ctk.CTkFont(size=11),
            fg_color="#3498db", hover_color="#2980b9",
            command=self._on_send_code_input,
        ).pack(side="right", padx=(0, 8), pady=6)
        # Enter in the entry submits too.
        self._code_input_entry.bind("<Return>", lambda e: self._on_send_code_input())

        # --- Live log (hidden by default; power users can toggle)
        # output_box is still the sink for every _append call. Cards render
        # the structured view by default; the raw log is just a fallback.
        self._log_visible = False
        self._log_toggle_btn = ctk.CTkButton(
            self.right_panel, text="Show Live Log",
            command=self._toggle_log, width=110, height=24,
            font=ctk.CTkFont(size=10), fg_color="transparent",
            text_color="#555555", hover_color="#333333")
        self._log_toggle_btn.pack(side="bottom", anchor="e", padx=8)

        self.output_box = ctk.CTkTextbox(self.right_panel,
                                          font=ctk.CTkFont(size=11),
                                          state="disabled", wrap="word",
                                          height=160)
        # Not packed -- _toggle_log packs/unpacks it on demand.

        # Kick off the Quick Ask inbox poller -- once per second on the main loop.
        self.root.after(1000, self._check_inbox)

    # --- Helpers ---

    def _on_mode_change(self, *args):
        """Update mode description label."""
        mode = self.mode_var.get()
        b = self.builder_model_var.get().split("-")[0].capitalize()
        c = self.challenger_model_var.get().split("-")[0].capitalize()
        a = self.auditor_model_var.get().split("-")[0].capitalize()
        descriptions = {
            "fast": f"{b} only (paste response, done)",
            "review": f"{b} → {c} (review, no audit)",
            "full": f"{b} → {c} → {a} (full ARC)",
        }
        self._mode_desc.set(descriptions.get(mode, ""))

    # --- Pipeline visualization state ---

    _PIPELINE_IDLE = ("\u25CB", "#444444", "#16213e", "#888888")    # open circle
    _PIPELINE_ACTIVE = ("\u25CF", "#3498db", "#1a2744", "#3498db")  # filled dot, blue
    _PIPELINE_DONE = ("\u2713", "#2ecc71", "#16213e", "#2ecc71")    # check, green
    _PIPELINE_ERROR = ("\u2717", "#e74c3c", "#2e1a1a", "#e74c3c")   # cross, red

    def _pipeline_apply(self, node, state):
        dot_char, dot_color, frame_color, label_color = state
        node['dot'].configure(text=dot_char, text_color=dot_color)
        node['frame'].configure(fg_color=frame_color)
        node['label'].configure(text_color=label_color)

    def _pipeline_set_active(self, phase_key, model_name=""):
        """Mark the given phase active; demote any other currently-active phase."""
        target = self._pipeline_nodes.get(phase_key)
        if target is None:
            return
        for key, node in self._pipeline_nodes.items():
            if key == phase_key:
                self._pipeline_apply(node, self._PIPELINE_ACTIVE)
            else:
                # Any previously-active (blue) node transitions to complete.
                if node['dot'].cget("text_color") == self._PIPELINE_ACTIVE[1]:
                    self._pipeline_apply(node, self._PIPELINE_DONE)
        if model_name:
            # Short label: trim paths, take token before first dash, clip.
            short = model_name.split("/")[-1].split("-")[0][:12]
            target['model_label'].configure(text=short)

    def _pipeline_set_complete(self, phase_key):
        """Mark a phase as complete."""
        node = self._pipeline_nodes.get(phase_key)
        if node is not None:
            self._pipeline_apply(node, self._PIPELINE_DONE)

    def _pipeline_set_error(self, phase_key):
        """Mark a phase as failed."""
        node = self._pipeline_nodes.get(phase_key)
        if node is not None:
            self._pipeline_apply(node, self._PIPELINE_ERROR)

    def _pipeline_reset(self):
        """Return every phase to the idle state (start of a new cycle)."""
        for node in self._pipeline_nodes.values():
            self._pipeline_apply(node, self._PIPELINE_IDLE)
            node['model_label'].configure(text="")

    def _on_mc_toggle(self, *args):
        """Hide the Monte Carlo pipeline node (and its trailing arrow) when off.

        Re-pack into the same slot when enabled so the node returns to the
        front of the chain.
        """
        mc_node = self._pipeline_nodes.get('mc')
        builder_node = self._pipeline_nodes.get('builder')
        if mc_node is None:
            return
        if self.mc_var.get():
            # Anchor both re-packs to builder's frame (always packed).
            # Pack mc first, then the arrow; `before=builder_node['frame']`
            # on side="left" inserts each widget immediately LEFT of builder,
            # yielding [..., mc, arrow, builder, ...].
            if builder_node is not None:
                mc_node['frame'].pack(side="left", padx=4, pady=8,
                                      before=builder_node['frame'])
                if builder_node['arrow_before'] is not None:
                    builder_node['arrow_before'].pack(
                        side="left", padx=2, before=builder_node['frame'])
            else:
                mc_node['frame'].pack(side="left", padx=4, pady=8)
        else:
            mc_node['frame'].pack_forget()
            # Also hide the arrow that would otherwise orphan in front of Builder.
            if builder_node and builder_node['arrow_before'] is not None:
                builder_node['arrow_before'].pack_forget()

    # --- Results rendering (verdict card + issue cards + raw panels) ---

    # --- Code-input row (shown only while Claude Code is waiting) ---

    def _show_code_input(self):
        """Reveal the input row above the status bar + focus the entry."""
        if self._code_input_frame.winfo_manager() != 'pack':
            self._code_input_frame.pack(side="bottom", fill="x",
                                         padx=8, pady=(0, 4))
        self._code_input_entry.focus_set()

    def _hide_code_input(self):
        """Hide the input row and clear the entry."""
        self._code_input_var.set("")
        self._code_input_frame.pack_forget()

    def _on_send_code_input(self):
        """Forward the entry text to the active JARVIS pipeline, if any."""
        text = self._code_input_var.get()
        pipeline = getattr(self, '_active_pipeline', None)
        if pipeline is None:
            return
        if pipeline.send_input(text):
            self._code_input_var.set("")
            # Hide now -- the pipeline already snapped state back to running,
            # and the poll loop will pick up any new output via get_new_output().
            self._hide_code_input()

    def _toggle_log(self):
        """Reveal or hide the raw live log textbox."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self.output_box.pack(side="bottom", fill="both", expand=False,
                                 padx=8, pady=(0, 4))
            self._log_toggle_btn.configure(text="Hide Live Log")
        else:
            self.output_box.pack_forget()
            self._log_toggle_btn.configure(text="Show Live Log")

    def _clear_results(self):
        """Wipe the verdict card, issues, and raw panels. Called on new cycle."""
        for frame in (self.verdict_frame, self.issues_frame, self.raw_panels_frame):
            for w in frame.winfo_children():
                w.destroy()
            frame.pack_forget()

    def _render_results(self):
        """Render verdict card + issues + raw panels on the main Tk thread.

        Safe to call from worker threads -- each sub-render schedules itself.
        """
        self.root.after(0, self._render_verdict)
        self.root.after(0, self._render_issues)
        self.root.after(0, self._render_raw_panels)

    def _render_verdict(self):
        """Top-of-results card: confidence + disagreement + exec summary + issue tally."""
        for w in self.verdict_frame.winfo_children():
            w.destroy()
        self.verdict_frame.pack(fill="x", padx=4, pady=(4, 8))

        confidence = "UNKNOWN"
        summary = ""
        if self._auditor_json:
            confidence = (self._auditor_json.get('confidence') or 'UNKNOWN').upper()
            summary = self._auditor_json.get('executive_summary', '')

        conf_colors = {
            'HIGH':   ("#2ecc71", "Safe to proceed"),
            'MEDIUM': ("#f39c12", "Proceed with modifications"),
            'LOW':    ("#e74c3c", "Review carefully"),
        }
        color, action = conf_colors.get(confidence, ("#888888", ""))

        conf_frame = ctk.CTkFrame(self.verdict_frame, fg_color="transparent")
        conf_frame.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(conf_frame, text=f"\u25CF {confidence} CONFIDENCE",
                      font=ctk.CTkFont(size=14, weight="bold"),
                      text_color=color).pack(side="left")
        if action:
            ctk.CTkLabel(conf_frame, text=f"  \u2014  {action}",
                          font=ctk.CTkFont(size=11),
                          text_color="#888888").pack(side="left")

        if self._disagreement and self._disagreement.get('classification'):
            da = self._disagreement
            da_icons = {
                'VALIDATED':    ('\u2705', '#2ecc71'),
                'EVOLVED':      ('\u21BB', '#3498db'),
                'DIVERGED':     ('\u26A1', '#f39c12'),
                'CONTRADICTED': ('\u26A0', '#e74c3c'),
            }
            icon, da_color = da_icons.get(da['classification'], ('\u2753', '#888888'))
            da_frame = ctk.CTkFrame(self.verdict_frame, fg_color="transparent")
            da_frame.pack(fill="x", padx=12, pady=(0, 4))
            ctk.CTkLabel(da_frame, text=f"{icon} {da['classification']}",
                          font=ctk.CTkFont(size=12),
                          text_color=da_color).pack(side="left")
            ctk.CTkLabel(da_frame, text=f"  {da.get('explanation', '')}",
                          font=ctk.CTkFont(size=10),
                          text_color="#888888", wraplength=460,
                          justify="left").pack(side="left", padx=(4, 0))

        if summary:
            ctk.CTkLabel(self.verdict_frame, text=summary,
                          font=ctk.CTkFont(size=12),
                          text_color="#cccccc", wraplength=500,
                          justify="left", anchor="w").pack(
                              fill="x", padx=12, pady=(4, 4))

        n = len(self._merged_issues)
        if n > 0:
            sev_counts = {}
            for issue in self._merged_issues:
                s = (issue.get('severity') or 'unknown').lower()
                sev_counts[s] = sev_counts.get(s, 0) + 1
            count_str = ", ".join(f"{v} {k}" for k, v in sorted(sev_counts.items()))
            ctk.CTkLabel(self.verdict_frame,
                          text=f"{n} issues found ({count_str})",
                          font=ctk.CTkFont(size=11),
                          text_color="#aaaaaa").pack(
                              fill="x", padx=12, pady=(0, 10))
        else:
            ctk.CTkLabel(self.verdict_frame,
                          text="No structured issues extracted",
                          font=ctk.CTkFont(size=11),
                          text_color="#666666").pack(
                              fill="x", padx=12, pady=(0, 10))

    def _render_issues(self):
        """One collapsible card per merged issue, severity-colored, click-to-expand."""
        for w in self.issues_frame.winfo_children():
            w.destroy()

        if not self._merged_issues:
            self.issues_frame.pack_forget()
            return

        self.issues_frame.pack(fill="x", padx=4, pady=(0, 8))

        sev_colors = {
            'critical': '#e74c3c',
            'high':     '#e67e22',
            'medium':   '#f39c12',
            'low':      '#3498db',
        }

        for issue in self._merged_issues:
            severity = (issue.get('severity') or 'unknown').lower()
            color = sev_colors.get(severity, '#888888')
            source = (issue.get('source') or 'unknown').title()

            card = ctk.CTkFrame(self.issues_frame, fg_color="#1a1a2e",
                                 corner_radius=8)
            card.pack(fill="x", pady=3)

            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=10, pady=(8, 4))

            ctk.CTkLabel(header, text="\u25CF",
                          font=ctk.CTkFont(size=10),
                          text_color=color).pack(side="left")
            ctk.CTkLabel(header, text=issue.get('title', 'Untitled'),
                          font=ctk.CTkFont(size=12, weight="bold"),
                          text_color="#dddddd").pack(side="left", padx=(6, 0))
            ctk.CTkLabel(header, text=f"{severity.upper()} \u2022 {source}",
                          font=ctk.CTkFont(size=9),
                          text_color=color).pack(side="right")

            detail = ctk.CTkFrame(card, fg_color="transparent")
            detail._visible = False

            desc = issue.get('description', '')
            suggestion = issue.get('suggestion', '')
            detail_text = desc
            if suggestion:
                detail_text += f"\n\nSuggestion: {suggestion}"
            ctk.CTkLabel(detail, text=detail_text,
                          font=ctk.CTkFont(size=11),
                          text_color="#aaaaaa", wraplength=450,
                          justify="left", anchor="nw").pack(
                              fill="x", padx=10, pady=(0, 8))

            def make_toggle(d=detail):
                def toggle(event=None):
                    if d._visible:
                        d.pack_forget()
                        d._visible = False
                    else:
                        d.pack(fill="x")
                        d._visible = True
                return toggle

            # Bind click to header AND each child so clicks anywhere in the
            # row toggle the detail, not just on whitespace.
            tog = make_toggle()
            header.bind("<Button-1>", tog)
            for child in header.winfo_children():
                child.bind("<Button-1>", tog)

    def _render_raw_panels(self):
        """Collapsible textboxes with the full raw responses (all closed by default)."""
        for w in self.raw_panels_frame.winfo_children():
            w.destroy()

        panels = []
        if self._builder_resp:
            panels.append(("Builder's Proposal", self._builder_resp))
        if self._gpt_resp:
            panels.append(("Challenger's Review", self._gpt_resp))
        if self._gemini_resp and not self._gemini_resp.startswith("("):
            panels.append(("Auditor's Audit", self._gemini_resp))
        if self._convergence:
            conv_text = "\n\n".join(
                f"**{k.title()}:** {v}" for k, v in self._convergence.items())
            panels.append(("Convergence Responses", conv_text))
        if self._mc_result and self._mc_result.get('all_analyses'):
            mc_text = "\n\n".join(
                f"Agent {i+1}: {a['analysis'][:500]}"
                for i, a in enumerate(self._mc_result['all_analyses']))
            panels.append((
                f"Monte Carlo ({len(self._mc_result['all_analyses'])} agents)",
                mc_text))

        if not panels:
            self.raw_panels_frame.pack_forget()
            return

        self.raw_panels_frame.pack(fill="x", padx=4, pady=(0, 8))
        for title, content in panels:
            self._add_collapsible_panel(title, content)

    def _add_collapsible_panel(self, title, content):
        """Single collapsed panel; clicking the header expands a CTkTextbox."""
        container = ctk.CTkFrame(self.raw_panels_frame, fg_color="#0d0d1a",
                                  corner_radius=6)
        container.pack(fill="x", pady=2)

        # Button acts as the header; starts collapsed (right-facing triangle).
        toggle_btn = ctk.CTkButton(
            container, text=f"\u25B8 {title}",
            fg_color="transparent", text_color="#888888",
            hover_color="#1a1a2e", anchor="w",
            font=ctk.CTkFont(size=11), height=28)
        toggle_btn.pack(fill="x", padx=4, pady=2)

        text_widget = ctk.CTkTextbox(container, font=ctk.CTkFont(size=11),
                                      height=200, state="disabled",
                                      fg_color="#0d0d1a")
        text_widget._visible = False

        def toggle():
            if text_widget._visible:
                text_widget.pack_forget()
                text_widget._visible = False
                toggle_btn.configure(text=f"\u25B8 {title}")
            else:
                text_widget.pack(fill="x", padx=8, pady=(0, 6))
                text_widget.configure(state="normal")
                text_widget.delete("1.0", "end")
                text_widget.insert("1.0", content)
                text_widget.configure(state="disabled")
                text_widget._visible = True
                toggle_btn.configure(text=f"\u25BE {title}")

        toggle_btn.configure(command=toggle)

    def _shuffle_roles(self):
        """Randomly shuffle which model plays which role."""
        import random
        models = [self.builder_model_var.get(),
                  self.challenger_model_var.get(),
                  self.auditor_model_var.get()]
        random.shuffle(models)
        self.builder_model_var.set(models[0])
        self.challenger_model_var.set(models[1])
        self.auditor_model_var.set(models[2])
        self._on_mode_change()  # refresh description

    def _get_gpt_prompt(self):
        """Return GPT's system prompt, with strict mode variant if enabled."""
        if self.strict_var.get():
            return (
                "You are the Adversary in ARC, a three-AI review system. "
                "Another AI (Claude) has proposed a solution to a problem. "
                "You MUST argue AGAINST Claude's solution. Find the fatal flaw. "
                "Assume the solution is wrong until proven otherwise. "
                "Challenge every assumption, question every design choice, and "
                "propose a fundamentally different approach. "
                "Do NOT agree with any part of the solution unless you have "
                "exhaustively tried to break it and failed. "
                "Be specific: name functions, cite line numbers if code is provided, "
                "and propose concrete alternatives for every issue you raise. "
                "This is a Red Team exercise — your job is to find the weakness."
            )
        return SYSTEM_PROMPTS['gpt']

    def _toggle_context(self):
        """Show/hide the Project Context field."""
        if self._context_visible:
            self.ctx_frame.pack_forget()
            self.ctx_toggle_btn.configure(text="▶ Project Context (optional)")
            self._context_visible = False
        else:
            self.ctx_frame.pack(fill="x", padx=12, pady=(2, 0),
                                after=self.ctx_toggle_btn.master)
            self.ctx_toggle_btn.configure(text="▼ Project Context (optional)")
            self._context_visible = True
            self.context_box.focus()

    def _toggle_roles(self):
        """Show/hide the Model Routing panel."""
        if self._roles_visible:
            self.role_frame.pack_forget()
            self.role_toggle_btn.configure(text="▶ Model Routing")
            self._roles_visible = False
        else:
            self.role_frame.pack(fill="x", padx=12, pady=(2, 0),
                                 after=self.role_toggle_btn.master)
            self.role_toggle_btn.configure(text="▼ Model Routing")
            self._roles_visible = True

    def _get_context(self):
        """Return project context text, or empty string if none."""
        return self.context_box.get("1.0", "end").strip()

    def _show_retry(self):
        """Show the retry button."""
        self.root.after(0, lambda: self.retry_btn.pack(side="left", padx=(8, 0)))

    def _hide_retry(self):
        """Hide the retry button."""
        self.root.after(0, lambda: self.retry_btn.pack_forget())

    def _on_retry(self):
        """Retry the failed pipeline phase and continue from there."""
        if not self._failed_phase or not self._pipeline_args:
            return
        self._hide_retry()
        self.run_btn.configure(state="disabled")
        phase = self._failed_phase
        self._failed_phase = None
        threading.Thread(
            target=self._pipeline_worker,
            args=self._pipeline_args,
            kwargs={"resume_from": phase},
            daemon=True).start()

    def _append(self, text):
        def _do():
            self.output_box.configure(state="normal")
            self.output_box.insert("end", text)
            self.output_box.see("end")
            self.output_box.configure(state="disabled")
        self.root.after(0, _do)

    def _clear_output(self):
        def _do():
            self.output_box.configure(state="normal")
            self.output_box.delete("1.0", "end")
            self.output_box.configure(state="disabled")
        self.root.after(0, _do)

    def _status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    # --- Auto-fill Claude via API (optional) ---

    def _on_autofill(self):
        problem = self.problem_box.get("1.0", "end").strip()
        if not problem:
            self._status("Enter a problem first.")
            return
        self.autofill_btn.configure(state="disabled")
        self._status("Calling Claude API...")
        threading.Thread(target=self._autofill_worker, args=(problem,), daemon=True).start()

    def _autofill_worker(self, problem):
        try:
            builder_model = self.builder_model_var.get()
            prompt = _with_context(problem, self._get_context())
            resp = call_any(builder_model, SYSTEM_PROMPTS['claude'], prompt)
            def _fill():
                self.claude_box.delete("1.0", "end")
                self.claude_box.insert("1.0", resp)
            self.root.after(0, _fill)
            self._status("Claude's response loaded. Click Run Review + Audit.")
        except Exception as e:
            self._status(f"Claude API error: {e}")
        finally:
            self.root.after(0, lambda: self.autofill_btn.configure(state="normal"))

    # --- Main pipeline: GPT review + Gemini audit ---

    def _on_run(self):
        if self._is_running:
            return  # prevent double execution
        problem = self.problem_box.get("1.0", "end").strip()
        claude_resp = self.claude_box.get("1.0", "end").strip()
        mode = self.mode_var.get()

        if not problem:
            self._status("Enter a problem first.")
            return
        if mode == "fast":
            # Fast mode: just show Claude's response (already pasted)
            if not claude_resp:
                self._status("Paste Claude's response — Fast mode just displays it.")
                return
            self._problem = problem
            self._claude_resp = claude_resp
            self._gpt_resp = "(Fast mode — no review)"
            self._gemini_resp = "(Fast mode — no audit)"
            self._clear_output()
            self._append("CLAUDE  (Builder)\n" + "─" * 40 + "\n")
            self._append(claude_resp + "\n\n")
            self._status("Fast mode — Claude's response displayed. Switch to Review or Full for critique.")
            self.save_btn.configure(state="normal")
            return
        if not claude_resp:
            self._status("Paste Claude's response first (or use Auto-fill).")
            return

        self._problem = problem
        self._claude_resp = claude_resp
        self._gpt_resp = ""
        self._gemini_resp = ""
        self._convergence = None
        self._disagreement = None
        self._mc_result = None
        self._challenger_json = None
        self._auditor_json = None
        self._merged_issues = []
        self._builder_resp = ""
        self._pipeline_reset()
        self._clear_results()
        self._clear_output()
        self.run_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.converge_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self._is_running = True

        threading.Thread(target=self._pipeline_worker,
                         args=(problem, claude_resp, self._get_context(), mode),
                         daemon=True).start()

    def _pipeline_worker(self, problem, claude_resp, context, mode, resume_from=None):
        sep = "\n" + "━" * 60 + "\n\n"
        strict = self.strict_var.get()
        challenger_system = self._get_gpt_prompt()
        challenger_model = self.challenger_model_var.get()
        auditor_model = self.auditor_model_var.get()

        # Capture the Builder's original response BEFORE Monte Carlo prepends
        # its merged framings. This is what the "Builder's Proposal" raw panel
        # shows; using the post-MC claude_resp would include MC's own analyses.
        if resume_from is None:
            self._builder_resp = claude_resp

        # --- Phase 0: Monte Carlo Divergence (optional, initial pass only) ---
        # Skipped when the checkbox is off, when resume_from is set (retries
        # must not re-run the fan-out -- that would change the Builder input
        # mid-pipeline and waste tokens), or when called via Fast/Review mode
        # where the extra framing doesn't feed anything downstream.
        if (resume_from is None
                and self.mc_var.get()
                and mode in ('review', 'full')):
            n = self.mc_n_var.get()
            if HAS_OLLAMA and OLLAMA_MODELS:
                mc_model = OLLAMA_MODELS[0]
            elif HAS_GPT:
                mc_model = "gpt-4o-mini"
            elif HAS_CLAUDE_API:
                mc_model = DEFAULT_CLAUDE_MODEL
            elif HAS_GEMINI:
                mc_model = DEFAULT_GEMINI_MODEL
            else:
                mc_model = None

            if mc_model is None:
                self._append("MONTE CARLO DIVERGENCE\n" + "─" * 40 + "\n")
                self._append("No model available for Monte Carlo -- skipping.\n\n")
            else:
                self._status(f"Monte Carlo: {n} parallel agents exploring the problem...")
                self._append("MONTE CARLO DIVERGENCE\n" + "─" * 40 + "\n")
                self._append(f"Running {n} parallel agents ({mc_model})...\n\n")
                self.root.after(0, self._pipeline_set_active, "mc", mc_model)

                try:
                    analyses = run_monte_carlo(problem, context, n=n, model=mc_model)
                    for i, a in enumerate(analyses, 1):
                        self._append(f"Agent {i}: {a['lens'][:60]}...\n")

                    self._status("Selecting top framings...")
                    self._mc_result = select_top_framings(problem, analyses, k=3)

                    self._append(f"\nSelected {len(self._mc_result['selected'])} framings.\n")
                    if self._mc_result['rejected_insights']:
                        self._append("Rejected insights preserved.\n")
                    self._append("\n")

                    # Prepend the merged framings to the Builder's response so
                    # the downstream Challenger+Auditor see both. (The Builder
                    # has already produced claude_resp by this point; injecting
                    # MC here is the path of least restructuring.)
                    if self._mc_result.get('merged_input'):
                        claude_resp = (self._mc_result['merged_input']
                                       + "\n---\n\n" + claude_resp)
                    self.root.after(0, self._pipeline_set_complete, "mc")
                except Exception as e:
                    self.root.after(0, self._pipeline_set_error, "mc")
                    self._append(f"Monte Carlo failed: {e}\n")
                    self._append("Continuing without divergence layer...\n\n")
                    self._mc_result = None

        # Store args for potential retry AFTER MC so the retry replays with
        # the same augmented claude_resp.
        self._pipeline_args = (problem, claude_resp, context, mode)

        # By the time the worker starts, the Builder's response is already in
        # hand (pasted or auto-filled). Mark that node complete so the chain
        # progresses visually. The builder_model_var carries the label.
        builder_model = self.builder_model_var.get()
        self.root.after(0, self._pipeline_set_active, "builder", builder_model)
        self.root.after(0, self._pipeline_set_complete, "builder")

        # --- Challenger Review ---
        if resume_from is None or resume_from == "gpt":
            role_label = "Adversary" if strict else "Challenger"
            self._status(f"{challenger_model} is {'attacking' if strict else 'reviewing'}...")
            if resume_from == "gpt":
                self._append("\n[Retrying Challenger...]\n\n")
            self._append(f"{challenger_model}  ({role_label})\n" + "─" * 40 + "\n")
            self.root.after(0, self._pipeline_set_active, "challenger", challenger_model)
            try:
                challenger_prompt = _with_context(
                    f"ORIGINAL PROBLEM:\n{problem}\n\n"
                    f"BUILDER'S PROPOSED SOLUTION:\n{claude_resp}\n\n"
                    f"Please review this solution.",
                    context)
                self._gpt_resp = call_any(challenger_model, challenger_system,
                                           challenger_prompt)
                self._append(self._gpt_resp + sep)
                # Pull the structured-output block out of the prose. None is
                # fine -- the UI just falls back to rendering the prose.
                self._challenger_json = extract_json_block(self._gpt_resp)
                self.root.after(0, self._pipeline_set_complete, "challenger")
            except Exception as e:
                self.root.after(0, self._pipeline_set_error, "challenger")
                self._gpt_resp = f"[ERROR] {e}"
                self._append(self._gpt_resp + "\n\n")
                self._append("⚠ Pipeline paused. Fix the issue and click Retry.\n")
                self._status("Challenger failed — pipeline paused. Click Retry when ready.")
                self._failed_phase = "gpt"
                self._is_running = False
                self._show_retry()
                self.root.after(0, lambda: self.run_btn.configure(state="normal"))
                return

        # --- Auditor (only in full mode) ---
        if resume_from is None or resume_from == "gemini":
            if mode == "full":
                self._status(f"{auditor_model} is auditing the exchange...")
                if resume_from == "gemini":
                    self._append("\n[Retrying Auditor...]\n\n")
                self._append(f"{auditor_model}  (Auditor + Synthesis)\n" + "─" * 40 + "\n")
                self.root.after(0, self._pipeline_set_active, "auditor", auditor_model)
                try:
                    auditor_prompt = _with_context(
                        f"ORIGINAL PROBLEM:\n{problem}\n\n"
                        f"BUILDER'S PROPOSED SOLUTION:\n{claude_resp}\n\n"
                        f"CHALLENGER'S REVIEW:\n{self._gpt_resp}\n\n"
                        f"Please audit this exchange.",
                        context)
                    self._gemini_resp = call_any(auditor_model,
                                                 SYSTEM_PROMPTS['gemini'],
                                                 auditor_prompt)
                    self._append(self._gemini_resp + "\n\n")
                    self._auditor_json = extract_json_block(self._gemini_resp)
                    self._merged_issues = merge_issues(
                        self._challenger_json, self._auditor_json)
                    self.root.after(0, self._pipeline_set_complete, "auditor")
                except Exception as e:
                    self.root.after(0, self._pipeline_set_error, "auditor")
                    self._gemini_resp = f"[ERROR] {e}"
                    self._append(self._gemini_resp + "\n\n")
                    self._append("⚠ Pipeline paused. Fix the issue and click Retry.\n")
                    self._status("Auditor failed — pipeline paused. Click Retry when ready.")
                    self._failed_phase = "gemini"
                    self._is_running = False
                    self._show_retry()
                    self.root.after(0, lambda: self.run_btn.configure(state="normal"))
                    return  # HALT
            else:
                self._gemini_resp = "(Review mode — no audit)"

        # Disagreement Analysis -- only runs when a Monte Carlo pass produced
        # parallel analyses. Without MC this block is a no-op, and the whole
        # feature is dormant until the Monte Carlo layer lands.
        if (self._mc_result and self._mc_result.get('all_analyses')
                and mode == "full" and not self._gemini_resp.startswith("[")):
            self._status("Analyzing agreement with initial analyses...")
            self._append("DISAGREEMENT ANALYSIS\n" + "─" * 40 + "\n")
            self.root.after(0, self._pipeline_set_active, "analysis")

            # Use the auditor's recommendation as the final output, appending
            # convergence notes if they ran (they usually run after this, so
            # typically convergence is None here -- kept for forward compat).
            final_output = self._gemini_resp
            if self._convergence:
                conv_text = ""
                for role, resp in self._convergence.items():
                    conv_text += f"{role}: {resp}\n"
                final_output = final_output + "\n\nCONVERGENCE:\n" + conv_text

            try:
                da = analyze_disagreement(
                    problem,
                    self._mc_result['all_analyses'],
                    final_output,
                )
                self._disagreement = da

                # Unicode escapes match the rest of arc.py's icon style.
                # \u2705 check  \u21BB cw-arrow  \u26A1 bolt  \u26A0 warning  \u2753 question
                icons = {
                    'VALIDATED':    '\u2705',
                    'EVOLVED':      '\u21BB',
                    'DIVERGED':     '\u26A1',
                    'CONTRADICTED': '\u26A0',
                }
                icon = icons.get(da['classification'], '\u2753')

                self._append(f"{icon} {da['classification']} "
                             f"(Confidence: {da['confidence']})\n\n")
                self._append(f"{da['explanation']}\n\n")

                self._status(f"ARC complete — {da['classification']} "
                             f"({da['confidence']} confidence)")
                self.root.after(0, self._pipeline_set_complete, "analysis")
            except Exception as e:
                self.root.after(0, self._pipeline_set_error, "analysis")
                self._append(f"Analysis failed: {e}\n\n")

        # Render the structured view once all phases are in.
        self.root.after(0, self._render_results)

        done_label = {"review": "Review", "full": "Full ARC"}
        self._status(f"{done_label.get(mode, 'ARC')} cycle complete.")
        self._is_running = False
        self.root.after(0, lambda: self.run_btn.configure(state="normal"))
        self.root.after(0, lambda: self.save_btn.configure(state="normal"))
        self.root.after(0, lambda: self.export_btn.configure(state="normal"))
        # Enable convergence + Execute if full mode completed with a real audit
        if mode == "full" and not self._gemini_resp.startswith("["):
            self.root.after(0, lambda: self.converge_btn.configure(state="normal"))
            self.root.after(0, lambda: self.execute_btn.configure(state="normal"))

    def _on_converge(self):
        """Send auditor's findings back to builder and challenger for agree/disagree."""
        if not hasattr(self, '_gemini_resp') or not self._gemini_resp:
            return
        if self._is_running:
            return
        self.converge_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")
        self._is_running = True
        self._status("Seeking convergence — Builder and Challenger responding to audit...")
        threading.Thread(target=self._convergence_worker, daemon=True).start()

    def _convergence_worker(self):
        sep = "\n" + "━" * 60 + "\n\n"
        context = self._get_context()
        builder_model = self.builder_model_var.get()
        challenger_model = self.challenger_model_var.get()
        self._convergence = {}

        self._append("─" * 60 + "\n")
        self._append("CONVERGENCE ROUND\n")
        self._append("─" * 60 + "\n\n")

        # Builder responds to audit
        self._status(f"{builder_model} (Builder) responding to audit...")
        self._append(f"{builder_model}  (Builder — Response to Audit)\n" + "─" * 40 + "\n")
        try:
            builder_conv = call_convergence(
                'builder', builder_model, self._problem,
                self._claude_resp, self._gpt_resp, self._gemini_resp,
                context=context)
            self._convergence['builder'] = builder_conv
            self._append(builder_conv + sep)
        except Exception as e:
            self._convergence['builder'] = f"[ERROR] {e}"
            self._append(f"[ERROR] {e}" + sep)

        # Challenger responds to audit
        self._status(f"{challenger_model} (Challenger) responding to audit...")
        self._append(f"{challenger_model}  (Challenger — Response to Audit)\n" + "─" * 40 + "\n")
        try:
            challenger_conv = call_convergence(
                'challenger', challenger_model, self._problem,
                self._claude_resp, self._gpt_resp, self._gemini_resp,
                context=context)
            self._convergence['challenger'] = challenger_conv
            self._append(challenger_conv + "\n\n")
        except Exception as e:
            self._convergence['challenger'] = f"[ERROR] {e}"
            self._append(f"[ERROR] {e}\n\n")

        self._status("Convergence complete — review agreements and disagreements above.")
        self._is_running = False
        self.root.after(0, lambda: self.run_btn.configure(state="normal"))

    def _on_save(self):
        if not hasattr(self, '_problem') or not self._problem:
            return
        fp = save_exchange(self._problem, self._claude_resp, self._gpt_resp,
                           self._gemini_resp, context=self._get_context(),
                           convergence=self._convergence,
                           disagreement=self._disagreement,
                           issues=self._merged_issues)
        self._status(f"Saved to {fp}")

    def _on_export_prompt(self):
        """Export the ARC cycle results as a Claude Code implementation prompt."""
        if not hasattr(self, '_problem') or not self._problem:
            return

        context = self._get_context()
        lines = []
        lines.append("TASK: [Describe the implementation task here]\n")
        if context:
            lines.append(f"Project Context: {context}\n")
        lines.append("---\n")
        lines.append("This solution was designed through an ARC review cycle ")
        lines.append("(Claude built, GPT challenged, Gemini audited).\n\n")

        # Extract key recommendations
        lines.append("PROBLEM:\n")
        lines.append(self._problem[:500])
        if len(self._problem) > 500:
            lines.append("...\n")
        lines.append("\n\n")

        lines.append("AGREED APPROACH (from ARC review):\n")
        # Use Gemini's executive recommendation if available, otherwise GPT's review
        if self._gemini_resp and not self._gemini_resp.startswith("[") and not self._gemini_resp.startswith("("):
            lines.append("[Gemini's Executive Recommendation — extract the actionable parts below]\n")
            # Take last 1000 chars of Gemini response (usually the recommendation)
            gemini_tail = self._gemini_resp[-1500:]
            lines.append(gemini_tail)
        elif self._gpt_resp and not self._gpt_resp.startswith("["):
            lines.append("[GPT's Review — extract the actionable parts below]\n")
            lines.append(self._gpt_resp[-1000:])
        lines.append("\n\n")

        # Add convergence insights if available
        if self._convergence:
            lines.append("CONVERGENCE NOTES:\n")
            if self._convergence.get('builder'):
                lines.append(f"Builder's response to audit:\n{self._convergence['builder'][:500]}\n\n")
            if self._convergence.get('challenger'):
                lines.append(f"Challenger's response to audit:\n{self._convergence['challenger'][:500]}\n\n")

        lines.append("---\n")
        lines.append("VERIFY:\n")
        lines.append("- All relevant files compile\n")
        lines.append("- Existing tests pass\n")
        lines.append("- New functionality works as described\n")

        prompt_text = "".join(lines)

        # Save to file
        save_dir = Path(__file__).parent / 'exports'
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fp = save_dir / f'prompt_{ts}.md'
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        # Also copy to clipboard
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(prompt_text)
            self._status(f"Prompt exported to {fp} and copied to clipboard")
        except Exception:
            self._status(f"Prompt exported to {fp}")

    # --- JARVIS execute pipeline ---

    def _on_execute(self):
        """Run the full JARVIS pipeline."""
        if self._is_running:
            return
        if not hasattr(self, '_gemini_resp') or not self._gemini_resp:
            self._status("Run a Full ARC cycle first.")
            return
        project_dir = self.project_dir_var.get().strip()
        if not project_dir or not os.path.isdir(project_dir):
            self._status("Set a valid project directory in Project Context.")
            return

        self._is_running = True
        self.execute_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")
        self._status("\u26A1 JARVIS Pipeline starting...")

        threading.Thread(target=self._execute_worker,
                         args=(project_dir,), daemon=True).start()

    def _execute_worker(self, project_dir):
        from pipeline import JarvisPipeline

        sep = "\n" + "\u2501" * 60 + "\n\n"

        # winsound is Windows-only. On other platforms the chimes silently
        # no-op; the status text + live output still update.
        try:
            import winsound
        except ImportError:
            winsound = None

        def _chime(alias):
            if not winsound:
                return
            try:
                winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
            except Exception:
                pass

        def on_code_status(status, detail):
            # Fires from a reader/watchdog thread inside pipeline.execute().
            # _append and _status marshal onto the Tk main thread via after().
            # UI-only work (show/hide input row) is scheduled explicitly.
            snippet = (detail or "")[:80]
            if status == "waiting":
                self._status(f"Code needs input: {snippet}")
                self._append(f"\n[CODE] Waiting for input: {detail}\n")
                self.root.after(0, self._show_code_input)
                _chime("SystemExclamation")
                self.root.after(500, lambda: _chime("SystemExclamation"))
            elif status == "running":
                # Snapped back to running (either input was sent or new data
                # arrived while we thought Code was idle). Hide the input row.
                self.root.after(0, self._hide_code_input)
            elif status == "finished":
                self._status("Code finished -- review changes")
                self._append("\n[CODE] Execution complete\n")
                self.root.after(0, self._hide_code_input)
                _chime("SystemAsterisk")
            elif status == "error":
                self._status(f"Code error: {snippet}")
                self._append(f"\n[CODE] Error: {detail}\n")
                self.root.after(0, self._hide_code_input)
                _chime("SystemHand")

        pipeline = JarvisPipeline(project_dir, on_status_change=on_code_status)
        # Expose the active pipeline so the Send button can reach it.
        self._active_pipeline = pipeline

        def _reenable():
            self._is_running = False
            self.root.after(0, lambda: self.run_btn.configure(state="normal"))
            self.root.after(0, lambda: self.execute_btn.configure(state="normal"))

        try:
            # CRITICAL: Check for dirty working tree BEFORE doing anything
            clean, msg = pipeline.check_clean_tree()
            if not clean:
                self._append(sep)
                self._append(f"\u26A0 PIPELINE ABORTED\n{msg}\n")
                self._status("Pipeline aborted: uncommitted changes. Commit or stash first.")
                _reenable()
                return

            # Step 1: Generate prompt
            self._append(sep)
            self._append("\u26A1 JARVIS PIPELINE\n" + "\u2500" * 40 + "\n\n")
            self._status("Generating Claude Code prompt...")

            prompt = pipeline.generate_prompt(self._problem, self._gemini_resp)
            self._append(f"Generated prompt ({len(prompt)} chars)\n\n")

            # Step 2: Create branch
            self._status("Creating isolated branch...")
            slug = self._problem[:30].strip()
            branch = pipeline.create_branch(slug)
            if branch:
                self._append(f"Branch: {branch}\n\n")
            else:
                self._append("Warning: Could not create branch\n\n")

            # Step 3: Execute Claude Code
            self._status("\u26A1 Claude Code is implementing changes...")

            # Live-poll the pipeline's new output every 2s while Code runs.
            # pipeline.execute() blocks this worker thread, but the Tk main
            # loop keeps firing root.after callbacks, so new lines stream
            # into the output panel as they appear.
            # Poll for new output every 1.5s. get_new_output() is cursor-
            # tracked so each chunk appears exactly once; no extra newline
            # because Code's stream already carries its own formatting.
            def _poll_output():
                status = pipeline.get_status()
                new = pipeline.get_new_output()
                if new:
                    self._append(new)
                if status in ("idle", "running", "waiting"):
                    self.root.after(1500, _poll_output)
            self.root.after(1500, _poll_output)

            success, output = pipeline.execute(prompt, budget_usd=1.0)

            if not success:
                self._append(f"Claude Code FAILED:\n{output}\n\n")
                self._status("Pipeline failed at execution step.")
                pipeline.revert()
                _reenable()
                return

            self._append("Claude Code finished.\n\n")

            # Step 4: Capture diff
            self._status("Capturing changes...")
            diff_stat, diff_full = pipeline.get_diff()

            if not diff_stat.strip():
                self._append("No files were changed.\n\n")
                pipeline.revert()
                self._status("No changes made. Branch cleaned up.")
                _reenable()
                return

            self._append(f"Changes:\n{diff_stat}\n")

            # Step 5: Run tests
            self._status("Running tests...")
            tests_passed, test_output = pipeline.run_tests()
            if tests_passed is True:
                self._append("Tests: PASSED \u2713\n\n")
            elif tests_passed is False:
                last_lines = test_output.strip().split('\n')[-5:]
                self._append(f"Tests: FAILED \u2717\n{chr(10).join(last_lines)}\n\n")
            else:
                self._append("Tests: could not run\n\n")

            # Step 6: Confidence check
            self._status("Auditor is rating the changes...")
            auditor_model = self.auditor_model_var.get()

            confidence_prompt = (
                "You are reviewing code changes that were ACTUALLY IMPLEMENTED "
                "based on your earlier recommendation.\n\n"
                f"ORIGINAL PROBLEM:\n{self._problem}\n\n"
                f"YOUR RECOMMENDATION (what should have been done):\n"
                f"{self._gemini_resp[:3000]}\n\n"
                f"GIT DIFF (what was actually changed):\n"
                f"{diff_full[:5000]}\n\n"
                f"TEST RESULTS:\n"
                f"{'PASSED' if tests_passed else 'FAILED or not run'}\n"
                f"{test_output[:500] if test_output else 'N/A'}\n\n"
                "Rate your confidence: HIGH / MEDIUM / LOW\n"
                "- HIGH: Changes match your recommendation, tests pass\n"
                "- MEDIUM: Mostly correct but has gaps or minor issues\n"
                "- LOW: Significant deviation or test failures\n\n"
                "Be specific: what was done well, what concerns you, "
                "and your final rating."
            )

            confidence_system = (
                "You are a QA auditor reviewing actual code changes against "
                "a plan you previously approved. Verify the implementation "
                "matches the intent. Be precise and direct."
            )

            confidence_resp = call_any(auditor_model, confidence_system,
                                       confidence_prompt)
            self._append("CONFIDENCE RATING\n" + "\u2500" * 40 + "\n")
            self._append(confidence_resp + "\n\n")

            # Step 7: Show confirm/revert
            self._append("\u2500" * 60 + "\n")
            self._append("ACTION REQUIRED:\n")
            self._append("Click COMMIT to keep changes, or REVERT to undo.\n\n")

            self._pipeline = pipeline
            self.root.after(0, self._show_pipeline_buttons)
            self._status("Review changes, then Commit or Revert.")
            # Leave execute_btn disabled until commit/revert resolves;
            # re-enable run_btn so the user can start new cycles.
            self._is_running = False
            self.root.after(0, lambda: self.run_btn.configure(state="normal"))

        except Exception as e:
            self._append(f"\n[ERROR] Pipeline failed: {e}\n\n")
            import traceback
            self._append(traceback.format_exc())
            self._status(f"Pipeline error: {e}")
            try:
                pipeline.revert()
            except Exception:
                pass
            _reenable()
        finally:
            # The Send button checks _active_pipeline; drop the reference so
            # stale send_input() calls can't reach a zombie subprocess.
            self._active_pipeline = None
            self.root.after(0, self._hide_code_input)

    def _show_pipeline_buttons(self):
        self.execute_btn.configure(state="disabled")
        if not hasattr(self, '_commit_btn'):
            self._commit_btn = ctk.CTkButton(
                self.execute_btn.master, text="\u2713 Commit",
                command=self._on_pipeline_commit,
                font=ctk.CTkFont(size=13),
                width=100, height=34,
                fg_color="#27AE60", hover_color="#1E8449")
        if not hasattr(self, '_revert_btn'):
            self._revert_btn = ctk.CTkButton(
                self.execute_btn.master, text="\u2717 Revert",
                command=self._on_pipeline_revert,
                font=ctk.CTkFont(size=13),
                width=100, height=34,
                fg_color="#C0392B", hover_color="#E74C3C")
        self._commit_btn.pack(side="left", padx=(8, 0))
        self._revert_btn.pack(side="left", padx=(4, 0))

    def _hide_pipeline_buttons(self):
        if hasattr(self, '_commit_btn'):
            self._commit_btn.pack_forget()
        if hasattr(self, '_revert_btn'):
            self._revert_btn.pack_forget()
        self.execute_btn.configure(state="normal")

    def _on_pipeline_commit(self):
        if not hasattr(self, '_pipeline') or not self._pipeline:
            return
        slug = self._problem[:50].strip()
        message = f"jarvis: {slug}"
        success = self._pipeline.commit(message)
        if success:
            self._append(f"Committed: {message}\n")
            self._status(f"Changes committed on {self._pipeline.branch_name}")
        else:
            self._append("Commit failed.\n")
        self._pipeline = None
        self._hide_pipeline_buttons()

    def _on_pipeline_revert(self):
        if not hasattr(self, '_pipeline') or not self._pipeline:
            return
        self._pipeline.revert()
        self._append("All changes reverted. Branch deleted.\n")
        self._status("Reverted -- codebase restored.")
        self._pipeline = None
        self._hide_pipeline_buttons()

    # --- Quick Ask inbox (IPC from Samsara) ---

    # Short spoken aliases -> full model IDs the API calls expect.
    # Without this, call_any("claude", ...) would hit the anthropic API
    # with literal model="claude" and 404.
    _QUICK_ASK_ALIASES = {
        'claude': DEFAULT_CLAUDE_MODEL,
        'gpt': DEFAULT_GPT_MODEL,
        'gemini': DEFAULT_GEMINI_MODEL,
    }

    def _check_inbox(self):
        """Poll the shared inbox file every 1 second for Quick Ask payloads."""
        try:
            if _ARC_INBOX.exists():
                with open(_ARC_INBOX, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                # Delete immediately so we don't read it twice.
                _ARC_INBOX.unlink(missing_ok=True)

                msg_type = payload.get("type", "")
                if msg_type == "quick_ask":
                    threading.Thread(
                        target=self._handle_quick_ask,
                        args=(payload,),
                        daemon=True,
                    ).start()
        except json.JSONDecodeError:
            # File being written at this exact moment -- catch next poll.
            pass
        except Exception as e:
            print(f"[INBOX] Error: {e}")

        # Re-queue -- runs every 1 second.
        self.root.after(1000, self._check_inbox)

    def _handle_quick_ask(self, payload):
        """Process a Quick Ask and display the result in the output panel."""
        model_alias = payload.get("model", "claude")
        question = payload.get("question", "")

        if not question:
            return

        resolved_model = self._QUICK_ASK_ALIASES.get(model_alias.lower(), model_alias)

        sep = "\n" + "\u2501" * 60 + "\n\n"
        self._append(sep)
        self._append(f"\U0001f3a4 QUICK ASK ({model_alias.upper()} -> {resolved_model})\n")
        self._append(f"{question}\n\n")
        self._status(f"Quick Ask: waiting for {model_alias}...")

        system_prompt = (
            "You are a concise technical assistant. Give a direct, "
            "helpful answer. No preamble, no fluff. If code is needed, "
            "include it. Keep answers short unless the question demands depth."
        )

        try:
            response = call_any(resolved_model, system_prompt, question)
            self._append(response + "\n\n")
            self._status("Quick Ask complete.")
        except Exception as e:
            self._append(f"Error: {e}\n\n")
            self._status(f"Quick Ask failed: {e}")

    def _save_preferences(self):
        """Save current UI state to config file."""
        config = {
            'mode': self.mode_var.get(),
            'strict_mode': self.strict_var.get(),
            'builder_model': self.builder_model_var.get(),
            'challenger_model': self.challenger_model_var.get(),
            'auditor_model': self.auditor_model_var.get(),
            'project_context': self._get_context(),
            'project_dir': self.project_dir_var.get(),
            'monte_carlo': self.mc_var.get(),
            'monte_carlo_n': self.mc_n_var.get(),
        }
        save_config(config)

    def _on_close(self):
        """Save preferences and close the app."""
        self._save_preferences()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("ARC — Adversarial Reasoning Chain")
    print(f"  Claude API: {'ready' if HAS_CLAUDE_API else 'manual mode (paste from claude.ai)'}")
    print(f"  GPT:        {'ready' if HAS_GPT else 'no key (set OPENAI_API_KEY in .env)'}")
    print(f"  Gemini:     {'ready' if HAS_GEMINI else 'no key (set GOOGLE_API_KEY in .env)'}")
    print(f"  Ollama:     {'ready (' + str(len(OLLAMA_MODELS)) + ' models)' if HAS_OLLAMA else 'not running'}")
    print()
    ARCApp().run()
