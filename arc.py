"""
ARC — Adversarial Reasoning Chain
Three-AI triangulated review: Claude builds, GPT challenges, Gemini audits.

Two modes:
  - Manual: paste Claude's response from claude.ai (no API key needed)
  - Auto: Claude API generates the response (requires ANTHROPIC_API_KEY)
GPT and Gemini always use APIs.
"""

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


def call_claude(problem, context="", model=None):
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or DEFAULT_CLAUDE_MODEL, max_tokens=4096,
        system=SYSTEM_PROMPTS['claude'],
        messages=[{"role": "user", "content": _with_context(problem, context)}],
    )
    return msg.content[0].text

def call_gpt(problem, claude_response, context="", system_prompt=None, model=None):
    client = openai.OpenAI()
    prompt = _with_context(
        f"ORIGINAL PROBLEM:\n{problem}\n\n"
        f"CLAUDE'S PROPOSED SOLUTION:\n{claude_response}\n\n"
        f"Please review Claude's solution.",
        context)
    resp = client.chat.completions.create(
        model=model or DEFAULT_GPT_MODEL, max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt or SYSTEM_PROMPTS['gpt']},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

def call_gemini(problem, claude_response, gpt_response, context="", model=None):
    m = genai.GenerativeModel(
        model_name=model or DEFAULT_GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPTS['gemini'],
    )
    prompt = _with_context(
        f"ORIGINAL PROBLEM:\n{problem}\n\n"
        f"CLAUDE'S PROPOSED SOLUTION:\n{claude_response}\n\n"
        f"GPT'S REVIEW:\n{gpt_response}\n\n"
        f"Please audit this exchange.",
        context)
    response = m.generate_content(prompt)

    # Handle empty/blocked responses gracefully
    try:
        return response.text
    except ValueError:
        # response.text throws ValueError if no valid parts
        # Gemini finish_reason enum: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
        reason = None
        if response.candidates and response.candidates[0].finish_reason:
            reason = response.candidates[0].finish_reason
        if reason == 3:
            return "[ERROR] Gemini safety filter blocked the response. Try rephrasing."
        elif reason == 1:
            return ("[ERROR] Model glitch: Gemini returned an empty successful response. "
                    "This is a known quirk with complex adversarial prompts. Try running again.")
        elif reason == 2:
            return "[ERROR] Gemini hit max token limit. Try a shorter input."
        elif reason == 4:
            return "[ERROR] Gemini blocked for recitation (copyright/IP). Try rephrasing."
        else:
            return f"[ERROR] Gemini returned no content. Finish reason: {reason}"


CONVERGENCE_PROMPT = (
    "You previously participated in an ARC review cycle. Here is the full exchange:\n\n"
    "ORIGINAL PROBLEM:\n{problem}\n\n"
    "CLAUDE'S SOLUTION:\n{claude_resp}\n\n"
    "GPT'S REVIEW:\n{gpt_resp}\n\n"
    "GEMINI'S AUDIT & RECOMMENDATION:\n{gemini_resp}\n\n"
    "Based on Gemini's audit and recommendation, respond with:\n"
    "1. AGREE or DISAGREE with each specific recommendation\n"
    "2. For each disagreement, explain WHY and propose an alternative\n"
    "3. Keep it concise — only address points where you have a strong opinion\n"
    "4. If you fully agree with everything, say so briefly and explain why"
)


def call_convergence_claude(problem, claude_resp, gpt_resp, gemini_resp,
                             context="", model=None):
    """Send Gemini's audit back to Claude for agree/disagree."""
    client = anthropic.Anthropic()
    prompt = CONVERGENCE_PROMPT.format(
        problem=problem, claude_resp=claude_resp,
        gpt_resp=gpt_resp, gemini_resp=gemini_resp)
    msg = client.messages.create(
        model=model or DEFAULT_CLAUDE_MODEL, max_tokens=4096,
        system="You are the Builder in ARC. You proposed a solution that was reviewed "
               "and audited. Now respond to the audit — agree or disagree with each point.",
        messages=[{"role": "user", "content": _with_context(prompt, context)}],
    )
    return msg.content[0].text


def call_convergence_gpt(problem, claude_resp, gpt_resp, gemini_resp,
                          context="", model=None):
    """Send Gemini's audit back to GPT for agree/disagree."""
    client = openai.OpenAI()
    prompt = CONVERGENCE_PROMPT.format(
        problem=problem, claude_resp=claude_resp,
        gpt_resp=gpt_resp, gemini_resp=gemini_resp)
    resp = client.chat.completions.create(
        model=model or DEFAULT_GPT_MODEL, max_tokens=4096,
        messages=[
            {"role": "system", "content":
             "You are the Reviewer in ARC. You reviewed a solution that was then audited "
             "by a third AI. Now respond to the audit — agree or disagree with each point."},
            {"role": "user", "content": _with_context(prompt, context)},
        ],
    )
    return resp.choices[0].message.content


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
            if convergence.get('claude'):
                f.write(f"### Claude's Response to Audit\n\n{convergence['claude']}\n\n")
            if convergence.get('gpt'):
                f.write(f"### GPT's Response to Audit\n\n{convergence['gpt']}\n\n")
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

        self._gpt_resp = ""
        self._gemini_resp = ""
        self._failed_phase = None  # 'gpt' or 'gemini' — set on API failure
        self._pipeline_args = None  # stored args for retry
        self._convergence = None  # {'claude': ..., 'gpt': ...} after convergence

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
        self.context_box.insert("1.0", "")
        self.context_box.configure(
            text_color="#666666")

        # --- Mode selector + Strict Mode ---
        mode_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        mode_frame.pack(fill="x", padx=12, pady=(8, 0))

        ctk.CTkLabel(mode_frame, text="Depth:",
                     font=ctk.CTkFont(size=12),
                     text_color="#888888").pack(side="left", padx=(4, 6))

        self.mode_var = ctk.StringVar(value="full")
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
        self.strict_var = ctk.BooleanVar(value=False)
        self.strict_check = ctk.CTkCheckBox(
            mode_frame, text="Strict Mode",
            variable=self.strict_var,
            font=ctk.CTkFont(size=12),
            height=28, checkbox_width=18, checkbox_height=18)
        self.strict_check.pack(side="right", padx=(0, 4))

        # --- Model selectors ---
        model_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        model_frame.pack(fill="x", padx=12, pady=(4, 0))

        ctk.CTkLabel(model_frame, text="Models:",
                     font=ctk.CTkFont(size=11),
                     text_color="#888888").pack(side="left", padx=(4, 6))

        # GPT model
        ctk.CTkLabel(model_frame, text="GPT:",
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")
        self.gpt_model_var = ctk.StringVar(value=DEFAULT_GPT_MODEL)
        ctk.CTkComboBox(model_frame, variable=self.gpt_model_var,
                        values=GPT_MODELS, width=120, height=24,
                        font=ctk.CTkFont(size=10),
                        state="readonly").pack(side="left", padx=(2, 10))

        # Gemini model
        ctk.CTkLabel(model_frame, text="Gemini:",
                     font=ctk.CTkFont(size=11),
                     text_color="#666666").pack(side="left")
        self.gemini_model_var = ctk.StringVar(value=DEFAULT_GEMINI_MODEL)
        ctk.CTkComboBox(model_frame, variable=self.gemini_model_var,
                        values=GEMINI_MODELS, width=140, height=24,
                        font=ctk.CTkFont(size=10),
                        state="readonly").pack(side="left", padx=(2, 10))

        # Claude model (only if API available)
        if HAS_CLAUDE_API:
            ctk.CTkLabel(model_frame, text="Claude:",
                         font=ctk.CTkFont(size=11),
                         text_color="#666666").pack(side="left")
            self.claude_model_var = ctk.StringVar(value=DEFAULT_CLAUDE_MODEL)
            ctk.CTkComboBox(model_frame, variable=self.claude_model_var,
                            values=CLAUDE_MODELS, width=180, height=24,
                            font=ctk.CTkFont(size=10),
                            state="readonly").pack(side="left", padx=(2, 0))
        else:
            self.claude_model_var = ctk.StringVar(value=DEFAULT_CLAUDE_MODEL)

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

    # --- Helpers ---

    def _on_mode_change(self, *args):
        """Update mode description label."""
        mode = self.mode_var.get()
        descriptions = {
            "fast": "Claude only (paste response, done)",
            "review": "Claude → GPT (review, no audit)",
            "full": "Claude → GPT → Gemini (full ARC)",
        }
        self._mode_desc.set(descriptions.get(mode, ""))

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
            # Pack it right after the toggle button, before the problem frame
            self.ctx_frame.pack(fill="x", padx=12, pady=(2, 0),
                                after=self.ctx_toggle_btn.master)
            self.ctx_toggle_btn.configure(text="▼ Project Context (optional)")
            self._context_visible = True
            self.context_box.focus()

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
            resp = call_claude(problem, context=self._get_context(),
                               model=self.claude_model_var.get())
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

        threading.Thread(target=self._pipeline_worker,
                         args=(problem, claude_resp, self._get_context(), mode),
                         daemon=True).start()

    def _pipeline_worker(self, problem, claude_resp, context, mode, resume_from=None):
        sep = "\n" + "━" * 60 + "\n\n"
        strict = self.strict_var.get()
        gpt_system = self._get_gpt_prompt()

        # Store args for potential retry
        self._pipeline_args = (problem, claude_resp, context, mode)

        # --- GPT Review ---
        if resume_from is None or resume_from == "gpt":
            if HAS_GPT:
                role_label = "Adversary" if strict else "Reviewer / Devil's Advocate"
                self._status(f"GPT is {'attacking' if strict else 'reviewing'} Claude's solution...")
                if resume_from == "gpt":
                    self._append("\n[Retrying GPT...]\n\n")
                self._append(f"GPT  ({role_label})\n" + "─" * 40 + "\n")
                try:
                    self._gpt_resp = call_gpt(problem, claude_resp,
                                               context=context, system_prompt=gpt_system,
                                               model=self.gpt_model_var.get())
                    self._append(self._gpt_resp + sep)
                except Exception as e:
                    self._gpt_resp = f"[ERROR] {e}"
                    self._append(self._gpt_resp + "\n\n")
                    self._append("⚠ Pipeline paused. Fix the issue and click Retry.\n")
                    self._status("GPT failed — pipeline paused. Click Retry when ready.")
                    self._failed_phase = "gpt"
                    self._show_retry()
                    self.root.after(0, lambda: self.run_btn.configure(state="normal"))
                    return  # HALT — do not continue to Gemini
            else:
                self._gpt_resp = "[SKIPPED — no OPENAI_API_KEY]"
                self._append("GPT  (Reviewer)\n" + "─" * 40 + "\n" + self._gpt_resp + sep)

        # --- Gemini Audit (only in full mode) ---
        if resume_from is None or resume_from == "gemini":
            if mode == "full":
                if HAS_GEMINI:
                    self._status("Gemini is auditing the exchange...")
                    if resume_from == "gemini":
                        self._append("\n[Retrying Gemini...]\n\n")
                    self._append("GEMINI  (Auditor + Synthesis)\n" + "─" * 40 + "\n")
                    try:
                        self._gemini_resp = call_gemini(problem, claude_resp, self._gpt_resp,
                                                        context=context, model=self.gemini_model_var.get())
                        self._append(self._gemini_resp + "\n\n")
                    except Exception as e:
                        self._gemini_resp = f"[ERROR] {e}"
                        self._append(self._gemini_resp + "\n\n")
                        self._append("⚠ Pipeline paused. Fix the issue and click Retry.\n")
                        self._status("Gemini failed — pipeline paused. Click Retry when ready.")
                        self._failed_phase = "gemini"
                        self._show_retry()
                        self.root.after(0, lambda: self.run_btn.configure(state="normal"))
                        return  # HALT
                else:
                    self._gemini_resp = "[SKIPPED — no GOOGLE_API_KEY]"
                    self._append("GEMINI  (Auditor)\n" + "─" * 40 + "\n" + self._gemini_resp + "\n\n")
            else:
                self._gemini_resp = "(Review mode — no audit)"

        done_label = {"review": "Review", "full": "Full ARC"}
        self._status(f"{done_label.get(mode, 'ARC')} cycle complete.")
        self.root.after(0, lambda: self.run_btn.configure(state="normal"))
        self.root.after(0, lambda: self.save_btn.configure(state="normal"))
        self.root.after(0, lambda: self.export_btn.configure(state="normal"))
        # Enable convergence if full mode completed with all three responses
        if mode == "full" and not self._gemini_resp.startswith("["):
            self.root.after(0, lambda: self.converge_btn.configure(state="normal"))

    def _on_converge(self):
        """Send Gemini's audit back to Claude and GPT for agree/disagree."""
        if not hasattr(self, '_gemini_resp') or not self._gemini_resp:
            return
        self.converge_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")
        self._status("Seeking convergence — Claude and GPT responding to audit...")
        threading.Thread(target=self._convergence_worker, daemon=True).start()

    def _convergence_worker(self):
        sep = "\n" + "━" * 60 + "\n\n"
        context = self._get_context()
        self._convergence = {}

        self._append("─" * 60 + "\n")
        self._append("CONVERGENCE ROUND\n")
        self._append("─" * 60 + "\n\n")

        # Claude responds to audit (parallel-ish but sequential for simplicity)
        if HAS_CLAUDE_API:
            self._status("Claude is responding to Gemini's audit...")
            self._append("CLAUDE  (Response to Audit)\n" + "─" * 40 + "\n")
            try:
                claude_conv = call_convergence_claude(
                    self._problem, self._claude_resp, self._gpt_resp,
                    self._gemini_resp, context=context,
                    model=self.claude_model_var.get())
                self._convergence['claude'] = claude_conv
                self._append(claude_conv + sep)
            except Exception as e:
                self._convergence['claude'] = f"[ERROR] {e}"
                self._append(f"[ERROR] {e}" + sep)
        else:
            self._convergence['claude'] = "(No Claude API — paste manually if needed)"
            self._append("CLAUDE  (Response to Audit)\n" + "─" * 40 + "\n")
            self._append(self._convergence['claude'] + sep)

        # GPT responds to audit
        if HAS_GPT:
            self._status("GPT is responding to Gemini's audit...")
            self._append("GPT  (Response to Audit)\n" + "─" * 40 + "\n")
            try:
                gpt_conv = call_convergence_gpt(
                    self._problem, self._claude_resp, self._gpt_resp,
                    self._gemini_resp, context=context,
                    model=self.gpt_model_var.get())
                self._convergence['gpt'] = gpt_conv
                self._append(gpt_conv + "\n\n")
            except Exception as e:
                self._convergence['gpt'] = f"[ERROR] {e}"
                self._append(f"[ERROR] {e}\n\n")
        else:
            self._convergence['gpt'] = "[SKIPPED — no OPENAI_API_KEY]"
            self._append("GPT  (Response to Audit)\n" + "─" * 40 + "\n")
            self._append(self._convergence['gpt'] + "\n\n")

        self._status("Convergence complete — review agreements and disagreements above.")
        self.root.after(0, lambda: self.run_btn.configure(state="normal"))
        self.root.after(0, lambda: self.converge_btn.configure(state="disabled"))

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
            if self._convergence.get('claude'):
                lines.append(f"Claude's response to audit:\n{self._convergence['claude'][:500]}\n\n")
            if self._convergence.get('gpt'):
                lines.append(f"GPT's response to audit:\n{self._convergence['gpt'][:500]}\n\n")

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

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("ARC — Adversarial Reasoning Chain")
    print(f"  Claude API: {'ready' if HAS_CLAUDE_API else 'manual mode (paste from claude.ai)'}")
    print(f"  GPT:        {'ready' if HAS_GPT else 'no key (set OPENAI_API_KEY in .env)'}")
    print(f"  Gemini:     {'ready' if HAS_GEMINI else 'no key (set GOOGLE_API_KEY in .env)'}")
    print()
    ARCApp().run()
