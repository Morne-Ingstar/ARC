# ARC — Adversarial Reasoning Chain

### Force AI models to disagree before they agree. Sequential adversarial review with Monte Carlo divergence — surfaces blind spots that consensus-driven systems miss.

---

## The Problem

Every major AI model has blind spots. When two AIs agree, it feels like consensus. Sometimes it's just shared ignorance. Running models in parallel generates diversity, not depth. You get three opinions, not three layers of scrutiny.

## The Approach

ARC uses **forced sequential critique with role-constrained reasoning**,preceded by an optional **Monte Carlo divergence layer** that explores the problem space from multiple angles before the chain begins:

```
Problem
  ↓
Monte Carlo (optional) — N parallel agents explore orthogonal framings
  ↓  Top 2-3 framings merged → fed into the chain
Builder — proposes a concrete solution
  ↓
Challenger — attacks the framing AND the solution
  ↓
Auditor — audits the exchange, finds consensus failures,
           delivers an Executive Recommendation
  ↓
Disagreement Analysis — classifies how the final output
           relates to the initial Monte Carlo analyses
  ↓
Governor (you) — reads the verdict card, reviews issue cards,
           clicks Refine & Re-run or ships it
```

Each AI's output becomes the input for the next. This creates **depth**, not just diversity.

## Quick Start

Requires **Python 3.10+** and at least two AI providers.

```bash
git clone https://github.com/Morne-Ingstar/ARC.git
cd ARC
cp .env.example .env
# Edit .env with your API keys (any 2 of 3)
pip install -r requirements.txt
python arc.py
```

**Providers supported:** Claude (Anthropic), GPT (OpenAI), Gemini (Google), Ollama (local models — free, no API key needed).

First launch shows a setup wizard that walks you through connecting your providers. You need at least two for adversarial review. Ollama alone works for experimentation.

**Cost per cycle:** \~$0.02–0.08 with cloud models. Free with Ollama.

## Features

### Monte Carlo Divergence Layer (v2)

Before the sequential chain runs, ARC sends the problem to N parallel agents (default: 6) with deliberately orthogonal reasoning lenses — first-principles analysis, adversarial thinking, simplicity-first, long-term maintainability, missing requirements, resource constraints, and more. 12 lenses in the pool, randomly sampled each run.

A selector picks the top 2-3 framings and merges them. Rejected framings' unique insights are preserved. The merged output feeds into the chain, breaking the "framing lock-in" problem where the Builder's initial frame determines everything downstream.

### Disagreement Analysis (v2)

After the chain completes, ARC compares the final recommendation against all Monte Carlo analyses and classifies the relationship:

- **VALIDATED** — chain confirmed what the parallel analyses found
- **EVOLVED** — chain built on initial analyses with significant improvements
- **DIVERGED** — chain found something the quick analyses missed (often good)
- **CONTRADICTED** — chain contradicts initial analyses without justification (flag for review)

This replaces naive vote-counting with reasoning-aware confidence scoring.

### Ollama Local Model Support (v2)

Any ARC role can use a local Ollama model. Auto-detects running models at startup. Ideal for Monte Carlo (cheap parallel exploration) and Builder (privacy-sensitive code). Challenger and Auditor benefit from stronger cloud models.

### Split-Panel IDE Layout (v2)

Left panel holds input and controls. Right panel shows the pipeline visualization and results. Problem stays visible while results appear — no context-switching.

### Pipeline Visualization (v2)

Horizontal pipeline shows exactly where you are:

```
[Monte Carlo] ──→ [Builder] ──→ [Challenger] ──→ [Auditor] ──→ [Analysis]
     ✓              ✓              ●                ○              ○
                  (Claude)       (GPT-5.4)
```

Each node shows its model, pulses when active, and checks green on completion. Monte Carlo node hides when disabled.

### Verdict Card + Issue Cards (v2)

Results are no longer a wall of text. The verdict card shows confidence (HIGH/MEDIUM/LOW with action guidance), disagreement classification, and executive summary. Below it: individual issue cards with severity dots (critical/high/medium/low), expandable on click. Full AI outputs are available in collapsible panels for power users.

### Refine & Re-run (v2)

Click the button and ARC pre-fills the problem with the original input plus all identified issues. Run again for a deeper review. Iteration counter tracks refinement depth. This turns ARC from a one-shot tool into an iterative feedback loop.

### First-Run Wizard (v2)

Three-step setup: welcome with explanation, API key status with auto-detected Ollama, and quick start. Only appears once.

### Core Features (v1)

**Three Depth Modes** — Fast (Builder only), Review (Builder → Challenger), Full (Builder → Challenger → Auditor).

**Strict Mode (Red Team)** — Forces the Challenger to argue against the solution. Must find the fatal flaw.

**Convergence Loop** — Sends the Auditor's findings back to Builder and Challenger. Each responds with agree/disagree. One round only.

**Role Rotation** — Any model can play any role. Shuffle button randomizes assignments.

**Project Context** — Define your domain once. Persists across sessions.

**Prompt Export** — Generates implementation prompts from cycle results.

**JARVIS Pipeline** — Turns recommendations into executed code via Claude Code CLI. Branch isolation, diff review, confidence rating, commit/revert.

**Quick Ask IPC** — Samsara voice integration for quick AI queries.

## When ARC Works (and When It Doesn't)

**Works best** for reasoning under uncertainty: architecture decisions,
code review, design tradeoffs, debugging strategies, risk assessment.

**Works poorly** for factual lookup. All three models can share the same
hallucination. ARC is a logic engine, not a search engine.

**Overkill** for simple tasks. Use Fast mode.

## Empirical Verification Gate

ARC v2.1 introduces a principle baked into all three role prompts: **no architectural recommendation may be based on an unverified assumption about external system behavior.**

This was learned the hard way. During development of Samsara (a voice control tool), ARC reviewed a problem where keyboard simulation failed while physical modifier keys were held. Three ARC rounds — involving Claude, GPT, and Gemini — debated Windows input architecture, held modifier key state, IME integration, and alternative hotkeys. All three models accepted the premise that "Windows re-asserts physical key state, making synthetic keystrokes unreliable" as an architectural fact.

**It wasn't.** A 10-line test script proved that the actual root cause was `pyautogui.hotkey()` silently dropping the Shift key from key combinations. The "fundamental Windows limitation" was a library bug. Three AI models, four iterations, and the fix was a one-function swap to `ctypes.SendInput`.

**The rule now:**

- **Builder:** When claiming an external system limitation, must provide a minimal test script (<30 lines) that proves it.
- **Challenger:** When the Builder claims "it can't be done," must demand the test script and propose an alternative test that could disprove the claim.
- **Auditor:** Must flag any claim about external system behavior that was accepted without empirical verification.

AI models are excellent at reasoning from premises but terrible at questioning whether the premises are true. The Empirical Verification Gate forces that question before reasoning begins.

## How ARC Differs

| Tool | Approach | Gap |
|------|----------|-----|
| **DSPy** (Stanford) | Framework for programming LLM pipelines | Optimizes pipelines, doesn't question reasoning quality |
| **GSD** | Project management layer for Claude Code | Organizes work, doesn't challenge AI decisions |
| **Tribunal** (MCP) | Three agents in parallel | No cross-critique, no depth |
| **Ixiom** | IDE with multi-agent coordinator | Cooperative agents, not adversarial |
| **ARC** | Sequential adversarial triad + Monte Carlo divergence | Each critiques the previous; divergence analysis validates the chain |

ARC occupies a unique position: it's not a framework, not a project
manager, and not an IDE. It's a **reasoning quality layer** — it
pressure-tests whether the thinking behind a solution is sound.

---

## The Story

ARC was born, designed, reviewed, and refined in a single development
session — and the tool was used to build itself.

The developer noticed that routing problems between Claude and GPT
consistently caught real gaps. But sometimes both would agree on
something wrong. Adding Gemini as a third observer from a different
model family caught things neither flagged.

The MVP (346 lines) was submitted to its own process for review. GPT
and Gemini independently identified the same issues and disagreed on
others. Every feature in the roadmap was proposed by one AI, challenged
by another, and audited by a third.

The v2 architecture (Monte Carlo, Ollama, disagreement analysis) was
designed the same way — proposed, sent to GPT and Gemini for review,
consensus extracted, then implemented. Zero disagreements on the three
core upgrades (top-k merge, orthogonal variation, reasoning-aware
confidence scoring).

---

## Architecture

`arc.py` (~2,100 lines) + `pipeline.py` (~380 lines). CustomTkinter UI.
No frameworks, no agents, no orchestration libraries — just a `call_any()`
router, a Monte Carlo fan-out with ThreadPoolExecutor, and a pipeline
that chains outputs sequentially.


## Roadmap

### Completed
- [x] Sequential pipeline, Strict Mode, Mode selector
- [x] Project Context, Exchange saving, Executive Recommendations
- [x] Halt on error + retry, Model selector, Role rotation
- [x] Convergence loop, Prompt export, Persistent config
- [x] JARVIS Pipeline (Execute → Claude Code → diff → confidence → Commit/Revert)
- [x] Quick Ask IPC (voice-triggered AI queries via Samsara)
- [x] Monte Carlo Divergence Layer (12 orthogonal lenses, top-k merge)
- [x] Ollama local model support (auto-detect, any role)
- [x] Disagreement Analysis (VALIDATED/EVOLVED/DIVERGED/CONTRADICTED)
- [x] Split-panel IDE layout with pipeline visualization
- [x] Structured JSON output + issue cards + verdict card
- [x] First-run setup wizard
- [x] Refine & Re-run iterative loop

### Planned
- [ ] JARVIS Active Orchestrator (unbuffered I/O, stdin, live monitoring)
- [ ] Cost tracking per session
- [ ] Session history with diff tracking across iterations
- [ ] Mobile companion app (phone as wireless mic for Samsara → ARC)

## The Name

**Adversarial** — the AIs challenge each other, not collaborate.
**Reasoning** — works on judgment under uncertainty, not factual lookup.
**Chain** — each link depends on the previous; depth, not collection.

## License

BSL-1.1 (Business Source License) — free for all non-commercial use.
Converts to MIT on April 23, 2030. See [LICENSE](LICENSE) for details.

If ARC's adversarial review process has improved your work, consider
[supporting the project](https://ko-fi.com/morneingstar) — built to make
AI reasoning trustworthy, kept free for the community.

## Acknowledgments

Built by [Morne](https://github.com/Morne-Ingstar). Every feature was
designed collaboratively with Claude (Anthropic), ChatGPT (OpenAI), and
Gemini (Google DeepMind) — and reviewed through the ARC process before
implementation.

---

<p align="center">
  <i>ARC was built in one session. It has been designing itself ever since.</i>
</p>
