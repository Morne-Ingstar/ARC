# ARC — Adversarial Reasoning Chain

### Multi-AI triangulated review with a human governor

---

## The Problem

Every major AI model has blind spots. Ask Claude a question, you get one
perspective shaped by one training set. Ask GPT the same question, you get
a different perspective — but often with the same underlying assumptions.
When two AIs agree, it feels like consensus. Sometimes it's just shared
ignorance.

Researchers call this the "shared blind spot" problem. When multiple LLMs
are trained on overlapping data with similar RLHF objectives, they develop
similar failure modes. They hallucinate the same fake libraries. They
overlook the same edge cases. They validate each other's mistakes.

The standard approach — running multiple AIs in parallel and comparing
outputs — doesn't solve this. Parallel comparison generates diversity,
not depth. You get three opinions, not three layers of scrutiny.

## The Idea

ARC takes a different approach: **forced sequential critique with
role-constrained reasoning.**

```
Problem
  ↓
Claude (Builder) — proposes a concrete solution
  ↓
GPT (Challenger) — forced to find weaknesses in Claude's solution
  ↓
Gemini (Auditor) — audits the exchange, finds what BOTH missed,
                   delivers an Executive Recommendation
  ↓
You (Governor) — reads all three, decides what to do
```

Each AI's output becomes the input for the next. GPT doesn't just give its
own opinion — it's forced to critique Claude's specific proposal. Gemini
doesn't just add a third voice — it's looking for "consensus failures"
where Claude and GPT agreed on something wrong.

This creates **depth**, not just diversity.

## Why It's Different

| Approach | How it works | What you get |
|----------|-------------|--------------|
| Single AI | Ask one model | One perspective |
| Parallel comparison | Ask three models the same question | Three opinions |
| **ARC** | Each model critiques the previous one's output | Three layers of scrutiny |

Parallel tools like MultipleChat send one prompt to multiple AIs and show
results side by side. Sequential tools like Helix AI Studio pass output
through a pipeline where each model contributes. ARC does neither — it
forces **adversarial review**, where each AI's job is to find what the
previous one got wrong.

In independent reviews of the ARC concept:
- **GPT** identified it as "role-constrained, sequential critique with
  enforced perspective separation" — not just multiple AIs talking, but a
  forced dependency chain that creates reasoning depth
- **Gemini** confirmed the sequential approach is "exactly what makes ARC
  valuable and differentiates it from tools like MultipleChat"
- Both independently recommended adding "Strict Mode" — a toggle that
  forces GPT to argue *against* Claude's solution, turning ARC into an
  automated Red Team

## The Name

**ARC** stands for **Adversarial Reasoning Chain**.

- **Adversarial** — the AIs don't collaborate, they challenge each other
- **Reasoning** — it works on problems requiring judgment under uncertainty,
  not factual lookup (ARC is a logic engine, not a search engine)
- **Chain** — each link depends on the previous one; the output is a chain
  of reasoning, not a collection of opinions

## Features

### Three Depth Modes

| Mode | Pipeline | When to use |
|------|----------|-------------|
| **Fast** | Claude only | Simple questions, quick drafts |
| **Review** | Claude → GPT | Design decisions, code review |
| **Full** | Claude → GPT → Gemini | Architecture, critical decisions |

### Strict Mode (Red Team)

A toggle that transforms GPT from a reviewer into an adversary. In Strict
Mode, GPT is instructed to argue *against* Claude's solution and find the
fatal flaw. It cannot agree unless it has exhaustively tried to break the
proposal and failed.

### Project Context

A persistent field where you define your domain once ("Python 3.11, Windows,
CustomTkinter, accessibility app"). Gets prepended to all three AI calls so
you don't repeat yourself every cycle.

### Human Governor

ARC deliberately keeps the human in the loop. The Governor reads all
perspectives and makes the final call. This prevents hallucination cascades
(where AIs validate each other's mistakes) and confirmation bias (where the
system just tells you what you want to hear).

### Exchange History

Every ARC cycle is saved as a timestamped markdown file in `exchanges/`,
preserving the full problem → solution → critique → audit chain for
reference.

### Model Selector

Dropdown per role lets you pick the exact model version:
- **GPT:** gpt-4o, gpt-4o-mini, o3, o3-mini
- **Gemini:** gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash
- **Claude:** claude-sonnet-4, claude-opus-4 (if API key set)

Swap models without editing code. When a model gets deprecated, just
pick a different one from the dropdown.

### Convergence Loop

After a Full ARC cycle, click **"Seek Convergence"** to send Gemini's
audit back to Claude and GPT. Each responds with agree/disagree on
specific recommendations. One round only — if they still disagree after
seeing the audit, that disagreement is the signal, not a bug.

### Prompt Export

After any cycle, click **"Export as Prompt"** to generate a ready-to-use
implementation prompt from the converged recommendations. Includes the
agreed approach, concerns to address, and project context. Copy directly
into Claude Code or any coding assistant.

## When ARC Works (and When It Doesn't)

**ARC works best** for problems requiring reasoning under uncertainty:
architecture decisions, code review, design tradeoffs, debugging strategies,
risk assessment — anywhere multiple valid approaches exist and the best
answer requires weighing tradeoffs.

**ARC works poorly** for factual lookup. If Claude hallucinates a fake Python
library, GPT might critique *how* Claude used it rather than questioning
whether it exists. All three models can share the same hallucination. For
factual verification, use a search-grounded tool like Perplexity.

**ARC is overkill** for simple tasks. You don't need a three-AI tribunal to
write a regex or center a div. Use Fast mode for those.

## Quick Start

### 1. Clone and set up API keys

```bash
git clone https://github.com/Morne-Ingstar/ARC.git
cd ARC
cp .env.example .env
```

Edit `.env` with your API keys:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

Claude API is **optional** — ARC is designed to work with your existing
claude.ai subscription. Paste Claude's response manually, or add an
`ANTHROPIC_API_KEY` for auto-fill.

### 2. Install dependencies

```bash
pip install customtkinter python-dotenv openai google-generativeai anthropic
```

### 3. Run

```bash
python arc.py
```

## How to Use

1. **Set Project Context** (optional) — click to expand, describe your domain
2. **Choose depth** — Fast, Review, or Full
3. **Toggle Strict Mode** if you want GPT to play adversary
4. **Type your problem** in the Problem field
5. **Paste Claude's response** (from claude.ai or any Claude conversation)
6. **Click "Run Review + Audit"** — GPT and Gemini process sequentially
7. **Read the output** — GPT's critique, then Gemini's audit + recommendation
8. **Save** — exports the full exchange as markdown

## Architecture

```
┌─────────────┐
│ You          │  Problem + Claude's response
│ (Governor)   │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│   GPT API    │  Reviews Claude's solution (or attacks it in Strict Mode)
│  (Challenger) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Gemini API  │  Audits the exchange + provides Executive Recommendation
│  (Auditor)   │
└──────────────┘
```

503 lines of Python. CustomTkinter UI. No frameworks, no agents, no
orchestration libraries — just three API calls in sequence with
carefully engineered system prompts.

## Research Context

ARC addresses a known problem in multi-agent AI systems that researchers
are actively debating how to solve.

**The problem with parallel multi-AI:** A 2026 research paper titled
"If You Want Coherence, Orchestrate a Team of Rivals" found that when AIs
run in parallel, the aggregating system "cannot detect conflicts it cannot
see." Differing AIs make conflicting implicit assumptions, and simply
merging their outputs doesn't resolve them. ARC's sequential approach —
where each AI reads and critiques the previous one's full output — is
the architecture these researchers theorize as the fix.

**What exists today and how ARC differs:**

| Tool | Approach | Limitation |
|------|----------|------------|
| **Tribunal** (MCP) | Three agents review code in parallel, synthesize a verdict | Agents can't read each other's analysis — no cross-critique |
| **MindStudio/Codex** | Claude calls GPT for a "second opinion" | Two-node system, no independent auditor |
| **AgentStack/Mastra** | Sequential cooperative pipeline (Architect → Reviewer → Tester) | Agents cooperate and pass a baton — not adversarial |
| **DeepTeam/Orq.ai** | Adversarial Red Team LLM attacks target LLM | Security testing only — not general reasoning |
| **ARC** | Sequential adversarial triad with meta-auditor | Each AI critiques the previous; third audits for consensus failures |

The key distinction: existing tools are either **parallel** (lacking
depth), **cooperative** (prone to sycophancy), or **adversarial but
narrow** (security-only). ARC combines sequential dependency with
adversarial role constraints and a meta-auditor — a combination that
is functionally unique in the current developer tooling landscape.

## Origin

ARC was born from a real workflow. While building
[Samsara](https://github.com/Morne-Ingstar/Samsara) (a voice dictation
app), the developer noticed that manually routing problems between Claude
and GPT consistently produced better solutions than using either alone.
GPT would catch gaps in Claude's proposals. But sometimes both would
share the same blind spot. Adding Gemini as an independent auditor —
from a different model family, with different training data — caught
things neither of them flagged.

The manual process (copy-pasting between chat windows) worked but was
tedious. ARC automates the routing while keeping the human as the final
decision-maker.

## API Costs

ARC uses frontier models for maximum quality. Approximate cost per
full cycle:

| Model | Role | Approximate cost |
|-------|------|-----------------|
| GPT-4o | Challenger | $0.01–0.05 |
| Gemini 2.0 Flash | Auditor | $0.01–0.03 |
| Claude (manual paste) | Builder | Free (your subscription) |
| **Total** | | **$0.02–0.08 per cycle** |

Fast mode is free. Review mode costs only the GPT call.

## Roadmap

- [x] Sequential pipeline (Claude → GPT → Gemini)
- [x] Strict Mode (Red Team)
- [x] Mode selector (Fast / Review / Full)
- [x] Project Context field
- [x] Exchange saving to markdown
- [x] Aggressive system prompts (Devil's Advocate + Consensus Failures)
- [x] Executive Recommendation in Gemini output
- [x] Halt on API error + retry button
- [x] Model selector per role (GPT, Gemini, Claude)
- [x] Convergence loop (Seek Convergence button)
- [x] Enum-based Gemini error diagnostics
- [x] Prompt export (Export as Prompt button)
- [ ] Role rotation (shuffle which model plays which role)
- [ ] Confidence divergence indicators
- [ ] Contradiction highlighting in UI
- [ ] Custom role definitions
- [ ] Local model support (Ollama)
- [ ] Cost tracking per session
- [ ] Voice input via Samsara integration

## License

MIT License — free for personal and commercial use.

## Acknowledgments

- Built by [Morne](https://github.com/Morne-Ingstar)
- Designed collaboratively with [Claude](https://anthropic.com) (Anthropic),
  [ChatGPT](https://openai.com) (OpenAI), and
  [Gemini](https://deepmind.google) (Google DeepMind)
- The concept itself was refined using the ARC process — three AIs
  reviewing each other's feedback on the tool that runs three-AI reviews

---

<p align="center">
  <i>ARC is powerful not because it uses multiple AIs — but because it
  forces them to reason against each other.</i>
</p>
