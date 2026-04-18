# ARC — Adversarial Reasoning Chain

### Multi-AI triangulated review with a human governor

---

## The Story

ARC was born, designed, reviewed, and refined in a single development
session — and the tool was used to build itself.

### The Observation

While building [Samsara](https://github.com/Morne-Ingstar/Samsara) (a
voice dictation app), the developer noticed a pattern: taking Claude's
proposed solution to GPT for a second opinion consistently caught real
gaps. GPT would find edge cases Claude missed. But sometimes both AIs
would agree on something that turned out to be wrong — same training
data, same blind spots.

Adding Gemini as a third observer from a different model family caught
things neither of them flagged. The manual workflow (copy-pasting between
three chat windows) was already valuable, but tedious.

### The MVP (346 lines)

The first version was simple: paste a problem, paste Claude's response,
click a button. GPT reviews Claude's output, Gemini audits both. Three
API calls in sequence, results displayed in a scrollable panel. It
worked, but it was rigid — hardcoded models, no error handling, no way
to control the depth of review.

### The ARC Review (the tool reviews itself)

The MVP was then submitted to Gemini for a full concept and code review.
Gemini's audit was sent to GPT for a meta-review. GPT's response went
back to Gemini. This three-way exchange — the exact process ARC
automates — produced the feature roadmap:

- **GPT** identified ARC as "role-constrained, sequential critique with
  enforced perspective separation" and proposed **Strict Mode** (forcing
  GPT to argue against Claude) and **Mode Selection** (Fast/Review/Full)
- **Gemini** called Strict Mode "the killer feature" and recommended
  **Project Context** (persistent domain info) and **Executive
  Recommendations** (structured synthesis in the audit output)
- **Both independently agreed** that the sequential pipeline was ARC's
  core differentiator from parallel comparison tools

Every feature that followed was designed through this same process.

### ARC in the Field (Samsara Development)

ARC was immediately put to work on real problems in
[Samsara](https://github.com/Morne-Ingstar/Samsara), the voice dictation
app that inspired it:

- **Echo cancellation** — Claude proposed an NLMS filter. After a Gemini
  API error, Claude confidently diagnosed "safety filter" as the cause.
  Gemini's audit corrected this — the finish_reason was STOP (1), not
  SAFETY (3). Claude had hallucinated the diagnosis. The fix was applied
  based on Gemini's correct reading of the API enum.
- **Speech threshold** — Claude proposed auto-calibration with a 0.01
  floor. Gemini's audit caught that 0.01 was the exact value that caused
  the original bug. Without the third perspective, the fix would have
  reintroduced the problem it was designed to solve.
- **Settings UI** — GPT recommended deleting the "pause" feature. Gemini
  pushed back: for a user with chronic hand pain, pausing mid-dictation
  without losing context is an accessibility lifeline, not a UX mess.
  The resulting 4-state machine was better than either AI's recommendation.

### ARC Reviews ARC

The tool was also submitted to its own process for a full code and
architecture review. GPT and Gemini independently identified the same
issues (brittle prefix-based routing, dead code, UI clutter, missing
persistent config) and disagreed on others (GPT rated Strict Mode high,
Gemini rated convergence higher). The resulting cleanup removed 60 lines
of dead code while adding persistent config and collapsible UI — changes
neither reviewer would have prioritized identically on their own.

### The Polished Product (975 lines)

By the end of the session, ARC had grown from 346 to 975 lines with
features designed, reviewed, and validated through its own process:

| Feature | Origin |
|---------|--------|
| Mode selector (Fast/Review/Full) | GPT's meta-review of Gemini's audit |
| Strict Mode (Red Team) | GPT proposed, Gemini validated as "killer feature" |
| Project Context | Gemini's initial review |
| Convergence Loop | Developer idea, refined through ARC cycle |
| Role Rotation | Developer idea, validated by both reviewers |
| Model Selector | Practical need (Gemini deprecated a model mid-session) |
| Halt on Error + Retry | Gemini's initial review |
| Prompt Export | Workflow observation (every cycle ended with writing a Code prompt) |
| Persistent Config | Both reviewers flagged as highest-priority quick win |
| Collapsible UI | Both reviewers flagged UI clutter independently |
| Registry Router | Both reviewers caught brittle prefix-based detection |
| Running Guard | GPT caught race condition on double-click |

The final code review (submitted to both GPT and Gemini simultaneously)
produced confidence ratings, identified remaining dead code, and
prioritized the roadmap — again, through the ARC process.

---

## The Problem ARC Solves

Every major AI model has blind spots. When two AIs agree, it feels like
consensus. Sometimes it's just shared ignorance.

The standard approach — running multiple AIs in parallel and comparing
outputs — doesn't solve this. Parallel comparison generates diversity,
not depth. You get three opinions, not three layers of scrutiny.

## The Approach

ARC uses **forced sequential critique with role-constrained reasoning:**

```
Problem
  ↓
Builder — proposes a concrete solution
  ↓
Challenger — forced to find weaknesses in the Builder's solution
  ↓
Auditor — audits the exchange, finds consensus failures,
           delivers an Executive Recommendation
  ↓
Governor (you) — reads all three, decides what to do
  ↓
[Optional] Seek Convergence — Builder and Challenger respond
           to the Auditor's findings (agree/disagree)
```

Each AI's output becomes the input for the next. The Challenger doesn't
just give its own opinion — it's forced to critique the Builder's
specific proposal. The Auditor doesn't just add a third voice — it's
looking for things both AIs agreed on that are actually wrong.

## Features

**Three Depth Modes** — Fast (Builder only), Review (Builder → Challenger),
Full (Builder → Challenger → Auditor). Don't burn three API calls on a
simple question.

**Strict Mode (Red Team)** — Toggle that forces the Challenger to argue
*against* the solution. Must find the fatal flaw. Cannot agree unless it
has exhaustively tried to break the proposal.

**Role Rotation** — Any model can play any role. Shuffle button randomizes
assignments. Test whether ARC's value comes from the structure or the
specific model assignments.

**Convergence Loop** — After a Full cycle, "Seek Convergence" sends the
Auditor's findings back to Builder and Challenger. Each responds with
agree/disagree. One round only — persistent disagreement is signal, not noise.

**Model Selector** — Dropdown per role showing all available models across
providers. Swap models without editing code. Collapsed behind a
"Model Routing" toggle to reduce UI clutter.

**Project Context** — Persistent field where you define your domain once.
Gets prepended to all AI calls. Saved across sessions.

**Prompt Export** — "Export as Prompt" generates a ready-to-use
implementation prompt from the converged recommendations. Copies to
clipboard and saves to `exports/`.

**Persistent Config** — Model selections, strict mode, project context
saved to `arc_config.json`. Everything persists across sessions.

**Halt on Error + Retry** — Pipeline pauses on API failure instead of
passing errors downstream. Red retry button re-runs only the failed phase.

**Enum Error Diagnostics** — Gemini errors show exact finish reason
(STOP/SAFETY/MAX_TOKENS/RECITATION) instead of generic error messages.

## Research Context

A 2026 research paper titled "If You Want Coherence, Orchestrate a Team
of Rivals" found that when AIs run in parallel, the aggregating system
"cannot detect conflicts it cannot see." ARC's sequential approach —
where each AI reads and critiques the previous one's full output — is
the architecture these researchers theorize as the fix.

**How ARC differs from existing tools:**

| Tool | Approach | Limitation |
|------|----------|------------|
| **Tribunal** (MCP) | Three agents review in parallel | No cross-critique between agents |
| **MindStudio/Codex** | Two-node second opinion | No independent auditor |
| **AgentStack/Mastra** | Cooperative sequential pipeline | Not adversarial |
| **DeepTeam/Orq.ai** | Adversarial Red Team | Security testing only |
| **ARC** | Sequential adversarial triad + meta-auditor | Each critiques the previous; third audits for consensus failures |

## When ARC Works (and When It Doesn't)

**Works best** for reasoning under uncertainty: architecture decisions,
code review, design tradeoffs, debugging strategies, risk assessment.

**Works poorly** for factual lookup. All three models can share the same
hallucination. ARC is a logic engine, not a search engine.

**Overkill** for simple tasks. Use Fast mode for those.

## Quick Start

```bash
git clone https://github.com/Morne-Ingstar/ARC.git
cd ARC
cp .env.example .env
# Edit .env with your API keys
pip install customtkinter python-dotenv openai google-generativeai anthropic
python arc.py
```

Claude API is **optional** — paste Claude's response manually from
claude.ai, or add an `ANTHROPIC_API_KEY` for auto-fill.

## Architecture

975 lines of Python. CustomTkinter UI. No frameworks, no agents, no
orchestration libraries — just a `call_any()` router that sends prompts
to the right provider based on a model registry, and a pipeline that
chains the outputs sequentially.

## API Costs

| Role | Default Model | Approximate cost |
|------|--------------|-----------------|
| Builder | Claude (manual paste) | Free (your subscription) |
| Challenger | GPT-4o | $0.01–0.05 |
| Auditor | Gemini 2.5 Flash | $0.01–0.03 |
| **Total** | | **$0.02–0.08 per cycle** |

Fast mode is free. Review mode costs only the Challenger call.

## Roadmap

- [x] Sequential pipeline
- [x] Strict Mode (Red Team)
- [x] Mode selector (Fast / Review / Full)
- [x] Project Context field
- [x] Exchange saving to markdown
- [x] Aggressive system prompts (Devil's Advocate + Consensus Failures)
- [x] Executive Recommendation in Auditor output
- [x] Halt on API error + retry button
- [x] Model selector per role
- [x] Role rotation + shuffle
- [x] Convergence loop
- [x] Prompt export
- [x] Enum-based error diagnostics
- [x] Persistent config
- [x] Collapsible model routing panel
- [x] Registry-based provider detection
- [x] Unified call_any router (zero dead code)
- [x] Running guard (prevents double execution)
- [ ] Confidence divergence indicators
- [ ] Disagreement summary layer
- [ ] Output formatting (markdown/structured)
- [ ] Local model support (Ollama)
- [ ] Cost tracking per session
- [ ] Voice input via Samsara integration

## The Name

**ARC** stands for **Adversarial Reasoning Chain**.

- **Adversarial** — the AIs don't collaborate, they challenge each other
- **Reasoning** — works on problems requiring judgment under uncertainty
- **Chain** — each link depends on the previous; output is a chain of
  reasoning, not a collection of opinions

## License

MIT License — free for personal and commercial use.

## Acknowledgments

- Built by [Morne](https://github.com/Morne-Ingstar)
- Designed collaboratively with [Claude](https://anthropic.com) (Anthropic),
  [ChatGPT](https://openai.com) (OpenAI), and
  [Gemini](https://deepmind.google) (Google DeepMind)
- Every feature was reviewed through the ARC process before implementation
- The concept itself was refined using ARC — three AIs reviewing each
  other's feedback on the tool that runs three-AI reviews

---

<p align="center">
  <i>ARC was built in one session. The tool designed itself through its
  own process. Every feature you see was proposed by one AI, challenged
  by another, and audited by a third.</i>
</p>
