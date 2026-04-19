# ARC — Adversarial Reasoning Chain

### Sequential adversarial AI review — forces reasoning depth through role-constrained critique chains, not parallel opinions.

---

## The Problem

Every major AI model has blind spots. When two AIs agree, it feels like
consensus. Sometimes it's just shared ignorance. Running models in
parallel generates diversity, not depth. You get three opinions, not
three layers of scrutiny.

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

Each AI's output becomes the input for the next. This creates **depth**,
not just diversity.

## Quick Start

Requires **Python 3.10+** and at least one API key (OpenAI or Google)
for the Challenger/Auditor roles.

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

**Cost per cycle:** ~$0.02–0.08 (Fast mode is free, Review is one API call).

## Features

**Three Depth Modes** — Fast (Builder only), Review (Builder → Challenger),
Full (Builder → Challenger → Auditor).

**Strict Mode (Red Team)** — Forces the Challenger to argue *against* the
solution. Must find the fatal flaw. Cannot agree unless it has exhaustively
tried to break the proposal.

**Role Rotation** — Any model can play any role. 🔀 Shuffle button
randomizes assignments. Test whether value comes from structure or model
assignments.

**Convergence Loop** — After a Full cycle, sends the Auditor's findings
back to Builder and Challenger. Each responds with agree/disagree. One
round only — persistent disagreement is signal, not noise.

**Model Selector** — Dropdown per role showing all available models.
Collapsed behind "▶ Model Routing" to keep the UI clean.

**Project Context** — Define your domain once ("Python 3.11, Windows,
accessibility app"). Persists across sessions.

**Prompt Export** — Generates a ready-to-use implementation prompt from
cycle results. Copies to clipboard + saves to `exports/`.

**Persistent Config** — Model selections, strict mode, project context
saved automatically. Everything persists across sessions.

**Halt on Error + Retry** — Pipeline pauses on API failure. Red retry
button re-runs only the failed phase.

## When ARC Works (and When It Doesn't)

**Works best** for reasoning under uncertainty: architecture decisions,
code review, design tradeoffs, debugging strategies, risk assessment.

**Works poorly** for factual lookup. All three models can share the same
hallucination. ARC is a logic engine, not a search engine.

**Overkill** for simple tasks. Use Fast mode.

---

## The Story

ARC was born, designed, reviewed, and refined in a single development
session — and the tool was used to build itself.

### The Observation

While building [Samsara](https://github.com/Morne-Ingstar/Samsara) (a
voice dictation app), the developer noticed that routing problems between
Claude and GPT consistently caught real gaps. But sometimes both would
agree on something wrong. Adding Gemini as a third observer from a
different model family caught things neither flagged.

### The MVP (346 lines)

Paste a problem, paste Claude's response, click a button. GPT reviews,
Gemini audits. Three API calls in sequence.

### The ARC Review (the tool designs itself)

The MVP was submitted to Gemini for a concept review. Gemini's audit
went to GPT for a meta-review. GPT's response went back to Gemini. This
three-way exchange produced the feature roadmap:

| Feature | Proposed by | Validated by |
|---------|------------|-------------|
| Strict Mode (Red Team) | GPT | Gemini ("killer feature") |
| Mode selector (Fast/Review/Full) | GPT | Gemini |
| Executive Recommendations | Gemini | GPT |
| Convergence Loop | Developer | Both reviewers |
| Persistent Config | Both reviewers | (unanimous) |

### ARC in the Field

ARC was immediately put to work on real problems in Samsara:

- **Echo cancellation** — Claude diagnosed a Gemini API error as "safety
  filter." Gemini's audit corrected this: the finish_reason was STOP (1),
  not SAFETY (3). Claude had hallucinated the diagnosis.
- **Speech threshold** — Claude proposed auto-calibration with a 0.01
  floor. Gemini caught that 0.01 was the exact value that caused the
  original bug. The fix would have reintroduced the problem it solved.
- **Settings UI** — GPT recommended deleting "pause." Gemini defended it
  as an accessibility lifeline. The resulting design was better than
  either recommendation alone.

### ARC Reviews ARC

The tool was submitted to its own process for a full code review. GPT
and Gemini independently identified the same issues (brittle routing,
dead code, UI clutter) and disagreed on others. The cleanup removed 60
lines of dead code while adding persistent config and collapsible UI.

### The Result (975 lines)

From 346-line MVP to 975-line production tool in one session. Every
feature was proposed by one AI, challenged by another, and audited by
a third.

---

## Research Context

A 2026 paper titled "If You Want Coherence, Orchestrate a Team of Rivals"
found that parallel AI systems "cannot detect conflicts" between agents.
ARC's sequential approach — where each AI critiques the previous one's
full output — is the architecture these researchers theorize as the fix.

| Tool | Approach | Limitation |
|------|----------|------------|
| **Tribunal** (MCP) | Three agents in parallel | No cross-critique |
| **MindStudio/Codex** | Two-node second opinion | No auditor |
| **AgentStack/Mastra** | Cooperative pipeline | Not adversarial |
| **DeepTeam/Orq.ai** | Adversarial Red Team | Security only |
| **ARC** | Sequential adversarial triad | Each critiques the previous; third audits for consensus failures |

## Architecture

975 lines of Python. CustomTkinter UI. No frameworks, no agents, no
orchestration libraries — just a `call_any()` router that sends prompts
to the right provider based on a model registry, and a pipeline that
chains the outputs sequentially.

## Roadmap

- [x] Sequential pipeline, Strict Mode, Mode selector
- [x] Project Context, Exchange saving, Executive Recommendations
- [x] Halt on error + retry, Model selector, Role rotation
- [x] Convergence loop, Prompt export, Persistent config
- [x] Collapsible UI, Registry router, Running guard
- [ ] Confidence divergence indicators
- [ ] Disagreement summary layer
- [ ] Output formatting (markdown/structured)
- [ ] Local model support (Ollama)
- [ ] Cost tracking per session
- [ ] Voice input via Samsara integration

## The Name

**Adversarial** — the AIs challenge each other, not collaborate.
**Reasoning** — works on judgment under uncertainty, not factual lookup.
**Chain** — each link depends on the previous; depth, not collection.

## License

MIT License — free for personal and commercial use.

If ARC's adversarial review process has improved your code, consider
[supporting the project](https://ko-fi.com/morneingstar) — built to make AI
collaboration smarter, kept free for the community.

## Acknowledgments

Built by [Morne](https://github.com/Morne-Ingstar). Designed
collaboratively with Claude (Anthropic), ChatGPT (OpenAI), and Gemini
(Google DeepMind). Every feature was reviewed through the ARC process
before implementation.

---

<p align="center">
  <i>ARC was built in one session. The tool designed itself through its
  own process.</i>
</p>
