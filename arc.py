"""
ARC — Adversarial Reasoning Chain
Three-AI triangulated review: Claude builds, GPT challenges, Gemini audits.

Two modes:
  - Manual: paste Claude's response from claude.ai (no API key needed)
  - Auto: Claude API generates the response (requires ANTHROPIC_API_KEY)
GPT and Gemini always use APIs.
"""

import json
import os
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
        "You are the Reviewer (Devil's Advocate) in ARC, a three-AI review system. "
        "Another AI (Claude) has proposed a solution to a problem. "
        "You will receive both the original problem and Claude's proposed solution. "
        "Your role: find weaknesses. Assume the solution has at least one significant "
        "flaw and find it. Look for architectural mistakes, missed edge cases, "
        "incorrect assumptions, scalability problems, and better alternatives. "
        "You MUST identify at least one concrete issue — do not simply validate "
        "the proposal. If the solution is genuinely excellent, explain exactly why "
        "each potential concern does not apply rather than just saying 'looks good.' "
        "Be specific: name functions, cite line numbers if code is provided, and "
        "propose concrete alternatives for every issue you raise."
    ),
    'gemini': (
        "You are the Auditor in ARC, a three-AI review system. "
        "Two AIs have weighed in: Claude proposed a solution, GPT reviewed it. "
        "You will receive the original problem, Claude's proposal, and GPT's review. "
        "Your role has TWO parts:\n\n"
        "PART 1 — AUDIT: Find what BOTH missed. Specifically look for:\n"
        "- Consensus Failures: things Claude and GPT happily agreed on that are "
        "actually wrong or suboptimal\n"
        "- Shared blind spots from similar training data\n"
        "- Missing requirements neither mentioned\n"
        "- Factual claims neither verified\n"
        "- Whether the agreed solution actually solves the original problem\n\n"
        "PART 2 — SYNTHESIS: Provide a clear 'Executive Recommendation' section "
        "at the end with:\n"
        "- What the Governor (human decision-maker) should actually DO\n"
        "- Which parts of Claude's solution to keep\n"
        "- Which parts to modify based on GPT's review\n"
        "- Any additional changes from your audit\n"
        "- A final confidence rating (High / Medium / Low) for the combined solution\n\n"
        "Do NOT repeat what Claude and GPT already said. Be the independent voice."
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

    return f"[ERROR] Unknown provider for model: {model_name}"


# All available models across all providers (for role assignment dropdowns)
ALL_MODELS = []
if HAS_CLAUDE_API:
    ALL_MODELS.extend(CLAUDE_MODELS)
if HAS_GPT:
    ALL_MODELS.extend(GPT_MODELS)
if HAS_GEMINI:
    ALL_MODELS.extend(GEMINI_MODELS)


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
    "4. If you fully agree with everything, say so briefly and explain why"
)

CONVERGENCE_SYSTEM = {
    'builder': ("You are the Builder in ARC. You proposed a solution that was reviewed "
                "and audited. Now respond to the audit — agree or disagree with each point."),
    'challenger': ("You are the Challenger in ARC. You reviewed a solution that was then audited "
                   "by a third AI. Now respond to the audit — agree or disagree with each point."),
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


def save_exchange(problem, claude_resp, gpt_resp, gemini_resp, context="",
                  convergence=None):
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
    return fp


# --- UI ---

class ARCApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("ARC")
        self.root.geometry("950x820")
        self.root.minsize(750, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load saved preferences
        self._config = load_config()

        self._gpt_resp = ""
        self._gemini_resp = ""
        self._failed_phase = None  # 'gpt' or 'gemini' — set on API failure
        self._pipeline_args = None  # stored args for retry
        self._convergence = None  # {'builder': ..., 'challenger': ...} after convergence
        self._is_running = False  # prevent double execution

        # --- Project Context (collapsible) ---
        self._context_visible = False
        ctx_toggle_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        ctx_toggle_frame.pack(fill="x", padx=12, pady=(10, 0))
        self.ctx_toggle_btn = ctk.CTkButton(
            ctx_toggle_frame, text="▶ Project Context (optional)",
            command=self._toggle_context, width=220, height=26,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            text_color="#888888", hover_color="#333333", anchor="w")
        self.ctx_toggle_btn.pack(side="left")

        self.ctx_frame = ctk.CTkFrame(self.root)
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
        mode_frame = ctk.CTkFrame(self.root, fg_color="transparent")
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

        # --- Role Assignment (collapsible) ---
        self._roles_visible = False
        role_toggle_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        role_toggle_frame.pack(fill="x", padx=12, pady=(4, 0))
        self.role_toggle_btn = ctk.CTkButton(
            role_toggle_frame, text="▶ Model Routing",
            command=self._toggle_roles, width=150, height=24,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            text_color="#888888", hover_color="#333333", anchor="w")
        self.role_toggle_btn.pack(side="left")

        self.role_frame = ctk.CTkFrame(self.root, fg_color="transparent")
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
        prob_frame = ctk.CTkFrame(self.root)
        prob_frame.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(prob_frame, text="Problem",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(fill="x", padx=8, pady=(6, 2))

        self.problem_box = ctk.CTkTextbox(prob_frame, height=70,
                                           font=ctk.CTkFont(size=13))
        self.problem_box.pack(fill="x", padx=8, pady=(0, 6))

        # --- Middle: Claude's response (paste or auto-fill) ---
        claude_frame = ctk.CTkFrame(self.root)
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
        btn_frame = ctk.CTkFrame(self.root, fg_color="transparent")
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
        for name, ready in [("GPT", HAS_GPT), ("Gemini", HAS_GEMINI)]:
            color = "#2ecc71" if ready else "#e74c3c"
            ctk.CTkLabel(dot_frame, text=f"● {name}",
                          font=ctk.CTkFont(size=11),
                          text_color=color).pack(side="left", padx=(8, 0))

        if HAS_CLAUDE_API:
            ctk.CTkLabel(dot_frame, text="● Claude API",
                          font=ctk.CTkFont(size=11),
                          text_color="#2ecc71").pack(side="left", padx=(8, 0))

        # --- Output: GPT + Gemini results ---
        self.output_box = ctk.CTkTextbox(self.root, font=ctk.CTkFont(size=13),
                                          state="disabled", wrap="word")
        self.output_box.pack(fill="both", expand=True, padx=12, pady=(4, 6))

        # --- Status bar ---
        self.status_var = ctk.StringVar(
            value="Paste Claude's response (or auto-fill), then click Run Review + Audit")
        ctk.CTkLabel(self.root, textvariable=self.status_var,
                      font=ctk.CTkFont(size=11),
                      text_color="#888888", anchor="w").pack(fill="x", padx=14, pady=(0, 8))

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

        # Store args for potential retry
        self._pipeline_args = (problem, claude_resp, context, mode)

        # --- Challenger Review ---
        if resume_from is None or resume_from == "gpt":
            role_label = "Adversary" if strict else "Challenger"
            self._status(f"{challenger_model} is {'attacking' if strict else 'reviewing'}...")
            if resume_from == "gpt":
                self._append("\n[Retrying Challenger...]\n\n")
            self._append(f"{challenger_model}  ({role_label})\n" + "─" * 40 + "\n")
            try:
                challenger_prompt = _with_context(
                    f"ORIGINAL PROBLEM:\n{problem}\n\n"
                    f"BUILDER'S PROPOSED SOLUTION:\n{claude_resp}\n\n"
                    f"Please review this solution.",
                    context)
                self._gpt_resp = call_any(challenger_model, challenger_system,
                                           challenger_prompt)
                self._append(self._gpt_resp + sep)
            except Exception as e:
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
                except Exception as e:
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
                           convergence=self._convergence)
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
        pipeline = JarvisPipeline(project_dir)

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
    print()
    ARCApp().run()
