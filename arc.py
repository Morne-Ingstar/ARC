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


def call_claude(problem, context=""):
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=4096,
        system=SYSTEM_PROMPTS['claude'],
        messages=[{"role": "user", "content": _with_context(problem, context)}],
    )
    return msg.content[0].text

def call_gpt(problem, claude_response, context="", system_prompt=None):
    client = openai.OpenAI()
    prompt = _with_context(
        f"ORIGINAL PROBLEM:\n{problem}\n\n"
        f"CLAUDE'S PROPOSED SOLUTION:\n{claude_response}\n\n"
        f"Please review Claude's solution.",
        context)
    resp = client.chat.completions.create(
        model="gpt-4o", max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt or SYSTEM_PROMPTS['gpt']},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

def call_gemini(problem, claude_response, gpt_response, context=""):
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPTS['gemini'],
    )
    prompt = _with_context(
        f"ORIGINAL PROBLEM:\n{problem}\n\n"
        f"CLAUDE'S PROPOSED SOLUTION:\n{claude_response}\n\n"
        f"GPT'S REVIEW:\n{gpt_response}\n\n"
        f"Please audit this exchange.",
        context)
    return model.generate_content(prompt).text


def save_exchange(problem, claude_resp, gpt_resp, gemini_resp, context=""):
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
            resp = call_claude(problem, context=self._get_context())
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
        self._clear_output()
        self.run_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")

        threading.Thread(target=self._pipeline_worker,
                         args=(problem, claude_resp, self._get_context(), mode),
                         daemon=True).start()

    def _pipeline_worker(self, problem, claude_resp, context, mode):
        sep = "\n" + "━" * 60 + "\n\n"
        strict = self.strict_var.get()
        gpt_system = self._get_gpt_prompt()

        # --- GPT Review ---
        if HAS_GPT:
            role_label = "Adversary" if strict else "Reviewer / Devil's Advocate"
            self._status(f"GPT is {'attacking' if strict else 'reviewing'} Claude's solution...")
            self._append(f"GPT  ({role_label})\n" + "─" * 40 + "\n")
            try:
                self._gpt_resp = call_gpt(problem, claude_resp,
                                           context=context, system_prompt=gpt_system)
                self._append(self._gpt_resp + sep)
            except Exception as e:
                self._gpt_resp = f"[ERROR] {e}"
                self._append(self._gpt_resp + sep)
        else:
            self._gpt_resp = "[SKIPPED — no OPENAI_API_KEY]"
            self._append("GPT  (Reviewer)\n" + "─" * 40 + "\n" + self._gpt_resp + sep)

        # --- Gemini Audit (only in full mode) ---
        if mode == "full":
            if HAS_GEMINI:
                self._status("Gemini is auditing the exchange...")
                self._append("GEMINI  (Auditor + Synthesis)\n" + "─" * 40 + "\n")
                try:
                    self._gemini_resp = call_gemini(problem, claude_resp, self._gpt_resp, context=context)
                    self._append(self._gemini_resp + "\n\n")
                except Exception as e:
                    self._gemini_resp = f"[ERROR] {e}"
                    self._append(self._gemini_resp + "\n\n")
            else:
                self._gemini_resp = "[SKIPPED — no GOOGLE_API_KEY]"
                self._append("GEMINI  (Auditor)\n" + "─" * 40 + "\n" + self._gemini_resp + "\n\n")
        else:
            self._gemini_resp = "(Review mode — no audit)"

        done_label = {"review": "Review", "full": "Full ARC"}
        self._status(f"{done_label.get(mode, 'ARC')} cycle complete.")
        self.root.after(0, lambda: self.run_btn.configure(state="normal"))
        self.root.after(0, lambda: self.save_btn.configure(state="normal"))

    def _on_save(self):
        if not hasattr(self, '_problem') or not self._problem:
            return
        fp = save_exchange(self._problem, self._claude_resp, self._gpt_resp,
                           self._gemini_resp, context=self._get_context())
        self._status(f"Saved to {fp}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("ARC — Adversarial Reasoning Chain")
    print(f"  Claude API: {'ready' if HAS_CLAUDE_API else 'manual mode (paste from claude.ai)'}")
    print(f"  GPT:        {'ready' if HAS_GPT else 'no key (set OPENAI_API_KEY in .env)'}")
    print(f"  Gemini:     {'ready' if HAS_GEMINI else 'no key (set GOOGLE_API_KEY in .env)'}")
    print()
    ARCApp().run()
