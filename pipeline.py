"""
JARVIS Pipeline -- ARC review -> Claude Code execution -> diff -> confidence.

Takes the Auditor's recommendation from an ARC cycle and:
1. Generates a Claude Code implementation prompt
2. Creates a jarvis/* git branch for isolation
3. Runs Claude Code CLI with --allowed-tools "Read Edit" (NO bash)
4. Captures git diff of what Code actually changed
5. Sends diff back to Auditor for confidence rating
6. Returns summary for user to confirm or revert
"""

import codecs
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# Substrings that signal Code is waiting for a human decision. Checked on an
# idle timer because interactive prompts like "Proceed? (y/n)" do NOT end in
# a newline -- line-buffered reads hang on them.
_WAITING_SUBSTRINGS = (
    ' allow ', ' deny ', 'y/n', 'yes/no',
    'permission', 'proceed?', 'continue?',
)

# If the tail of the buffer ends with one of these after a quiet period, we
# conclude Claude Code is waiting for input.
_PROMPT_TAILS = ('?', ':', '>')

# Idle threshold before we consider a non-newline-terminated tail a prompt.
_IDLE_SECONDS = 3.0

# Hard kill timeout (user can override in execute()).
_DEFAULT_MAX_RUNTIME = 300

# Strip ANSI terminal color codes from Claude CLI output
# (CLI outputs rich terminal formatting that renders as garbage in Tkinter)
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub('', text) if text else ''


class JarvisPipeline:

    def __init__(self, project_dir, on_status_change=None):
        self.project_dir = Path(project_dir)
        self.branch_name = None
        self.original_branch = None
        self.execution_log = None
        self.diff_stat = None
        self.diff_full = None
        self.test_output = None
        self.tests_passed = None

        # Live-monitoring state. Callback fires only on status TRANSITIONS
        # (running -> waiting -> finished/error) so the GUI doesn't re-chime
        # on every output chunk.
        self._on_status_change = on_status_change
        self.process = None

        # Character-level output buffer + cursor. The reader thread appends
        # raw decoded chunks under _output_lock; get_new_output() returns
        # just the tail past _output_cursor and advances it.
        self._output_buffer = ""
        self._output_cursor = 0
        self._output_lock = threading.Lock()

        # Timestamps for idle/timeout detection.
        self._start_time = None
        self._last_output_time = None

        self._code_status = "idle"
        self._last_reported_status = None
        self._reader_thread = None
        self._watchdog_thread = None
        # Event set by send_input() to snap the idle watchdog back to "running".
        self._input_sent = threading.Event()

    def generate_prompt(self, problem, auditor_recommendation):
        """Convert ARC output into a Claude Code prompt."""
        return (
            f"TASK:\n{problem}\n\n"
            f"APPROACH (designed through multi-AI adversarial review):\n"
            f"{auditor_recommendation}\n\n"
            f"CONSTRAINTS:\n"
            f"- Only modify files in this project directory\n"
            f"- Do not create new files unless the approach explicitly requires it\n"
            f"- Do not modify test files unless the approach explicitly requires it\n"
            f"- Leave all changes unstaged (do not run git commands)\n\n"
            f"VERIFY:\n"
            f"- All modified files pass syntax check (py_compile or equivalent)\n"
            f"- Changes match the approach described above\n"
        )

    def check_clean_tree(self):
        """Verify the git working tree is clean before proceeding.

        CRITICAL SAFETY CHECK: If there are uncommitted changes, the
        pipeline MUST NOT run. The revert step (git checkout . + git clean)
        would permanently destroy the user's uncommitted work.

        Returns:
            (clean: bool, message: str)
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True,
                cwd=self.project_dir
            )
            if result.stdout.strip():
                return False, (
                    "Cannot run pipeline: working directory is not clean.\n"
                    "Please commit or stash your changes first.\n\n"
                    "Dirty files:\n" + result.stdout
                )
            return True, "Clean"
        except FileNotFoundError:
            return False, "git not found"

    def create_branch(self, slug="change"):
        """Create a jarvis/* branch for isolation.

        Returns branch name, or None on failure.
        """
        try:
            self.original_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True,
                cwd=self.project_dir
            ).stdout.strip()

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Sanitize slug: only keep alphanumeric + hyphens
            # (git branch names cannot contain ?, *, [, \, spaces, trailing dots, etc.)
            safe_slug = re.sub(r'[^a-z0-9]', '-', slug.lower())[:30].strip('-')
            if not safe_slug:
                safe_slug = "change"
            self.branch_name = f"jarvis/{safe_slug}-{timestamp}"

            result = subprocess.run(
                ["git", "checkout", "-b", self.branch_name],
                cwd=self.project_dir, capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"[JARVIS] Branch creation failed: {result.stderr}")
                self.branch_name = None
                return None

            print(f"[JARVIS] Created branch: {self.branch_name}")
            return self.branch_name
        except FileNotFoundError:
            print("[JARVIS] git not found")
            return None

    def execute(self, prompt, budget_usd=1.25, timeout=_DEFAULT_MAX_RUNTIME):
        """Run Claude Code CLI and stream output character-by-character.

        SECURITY: --allowed-tools "Read Edit" restricts Code to file reading
        and editing ONLY. No bash, no shell commands. Enforced at CLI level.

        Three background concerns, three threads:
        - reader: unbuffered os.read on stdout, feeds the output buffer and
          a streaming UTF-8 decoder so split multi-byte sequences don't tear.
        - watchdog: polls last_output_time for idle + tail for prompt chars
          to detect 'waiting' state without needing a newline. Also kills
          the process if wall time exceeds `timeout`.
        - caller: this method, which blocks on process.wait() so downstream
          pipeline steps (diff, tests) only run after Code exits.

        Returns (success: bool, output: str). Output is ANSI-stripped.
        """
        with self._output_lock:
            self._output_buffer = ""
            self._output_cursor = 0
        self._last_reported_status = None
        self._start_time = time.time()
        self._last_output_time = self._start_time
        self._input_sent.clear()
        self._set_status("running")

        try:
            # bufsize=0 + text=False (binary) is the only combination that
            # guarantees unbuffered reads. text mode disallows bufsize=0
            # (Python raises ValueError). We decode chunks ourselves.
            self.process = subprocess.Popen(
                [
                    "claude", "-p", prompt,
                    "--print",
                    "--output-format", "text",
                    "--permission-mode", "auto",
                    "--allowed-tools", "Read Edit",
                    "--max-budget-usd", str(budget_usd),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                cwd=str(self.project_dir),
            )
        except FileNotFoundError:
            self.execution_log = "[JARVIS] 'claude' CLI not found. Install Claude Code first."
            self._set_status("error", self.execution_log)
            return False, self.execution_log

        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="jarvis-reader")
        self._reader_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, args=(timeout,),
            daemon=True, name="jarvis-watchdog")
        self._watchdog_thread.start()

        # Block this (worker) thread until the process exits. The reader and
        # watchdog handle state transitions + UI-visible progress in parallel.
        try:
            self.process.wait()
        except Exception:
            pass

        # Let the reader drain the last chunks now that the pipe is closed.
        self._reader_thread.join(timeout=2.0)

        with self._output_lock:
            clean_out = self._output_buffer
        self.execution_log = clean_out
        success = (self.process.returncode == 0)
        return success, clean_out

    def _set_status(self, status, detail=""):
        """Update status; fire callback ONLY on transitions."""
        self._code_status = status
        if status == self._last_reported_status:
            return
        self._last_reported_status = status
        if self._on_status_change is None:
            return
        try:
            self._on_status_change(status, detail)
        except Exception as e:
            # A callback bug should never kill the reader/watchdog.
            print(f"[JARVIS] status callback raised: {e}")

    def _read_loop(self):
        """Unbuffered stdout reader.

        Uses os.read on the raw fd so Python's TextIOWrapper doesn't do its
        own line buffering on top of bufsize=0. Decodes with an incremental
        UTF-8 decoder so multi-byte characters that straddle a chunk boundary
        (emojis, box-drawing) don't produce replacement chars.
        """
        fd = self.process.stdout.fileno()
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        try:
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break  # EOF
                text = decoder.decode(chunk)
                if not text:
                    continue
                clean = _strip_ansi(text)
                with self._output_lock:
                    self._output_buffer += clean
                self._last_output_time = time.time()
                # Any output while "waiting" means Code moved on -- snap back.
                if self._code_status == "waiting":
                    self._set_status("running")
        except Exception as e:
            print(f"[JARVIS] reader error: {e}")
        finally:
            # Flush any trailing bytes from the decoder.
            tail = decoder.decode(b'', final=True)
            if tail:
                with self._output_lock:
                    self._output_buffer += _strip_ansi(tail)

        # Pipe closed -> process is exiting. Get the exit code.
        try:
            rc = self.process.wait(timeout=5)
        except Exception:
            rc = -1
        terminal = "finished" if rc == 0 else "error"
        self._set_status(terminal, f"Exit code: {rc}")

    def _watchdog_loop(self, max_runtime):
        """Idle/prompt detector + wall-clock killer.

        Idle detection runs only while status is "running". If no output has
        arrived for _IDLE_SECONDS and the buffer tail ends with a prompt
        character (? : >) or contains a known waiting-substring, we transition
        to "waiting". The reader will transition back to "running" as soon as
        the next chunk arrives.
        """
        while self._code_status in ("running", "waiting"):
            time.sleep(0.5)

            # Wall-clock safety kill.
            if self.process is not None and self._start_time is not None:
                if time.time() - self._start_time > max_runtime:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                    self._set_status(
                        "error",
                        f"[JARVIS] Max runtime ({max_runtime}s) exceeded")
                    return

            if self._code_status != "running":
                continue
            if self._last_output_time is None:
                continue
            if time.time() - self._last_output_time < _IDLE_SECONDS:
                continue

            with self._output_lock:
                tail = self._output_buffer[-256:].rstrip()
            if not tail:
                continue
            lower = tail.lower()
            ends_with_prompt = tail[-1] in _PROMPT_TAILS
            has_waiting_word = any(p in lower for p in _WAITING_SUBSTRINGS)
            if ends_with_prompt or has_waiting_word:
                snippet = tail.splitlines()[-1] if tail.splitlines() else tail
                self._set_status("waiting", snippet[-120:])

    def send_input(self, text):
        """Write `text` + newline to the subprocess stdin and mirror it locally.

        Returns True if the write succeeded. Transitions the pipeline state
        back to "running" so downstream UI (input field visibility, etc.)
        immediately reflects that we're no longer waiting.
        """
        if self.process is None or self.process.stdin is None:
            return False
        if self.process.poll() is not None:
            return False  # already exited
        try:
            payload = (text + "\n").encode('utf-8')
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except Exception as e:
            print(f"[JARVIS] send_input failed: {e}")
            return False

        # Echo into the output buffer so the user sees what they sent.
        with self._output_lock:
            self._output_buffer += f"> {text}\n"
        self._last_output_time = time.time()
        self._input_sent.set()
        # Snap back to running immediately; the reader/watchdog will still
        # correctly flip to waiting again if the next prompt appears.
        if self._code_status == "waiting":
            self._set_status("running")
        return True

    def get_output(self):
        """Return the full output buffer (ANSI-stripped)."""
        with self._output_lock:
            return self._output_buffer

    def get_new_output(self):
        """Return text added since the last call; advance the cursor."""
        with self._output_lock:
            new = self._output_buffer[self._output_cursor:]
            self._output_cursor = len(self._output_buffer)
        return new

    def get_status(self):
        """Current Code status: idle | running | waiting | finished | error."""
        return self._code_status

    def get_diff(self):
        """Capture what Claude Code changed."""
        self.diff_stat = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True,
            cwd=self.project_dir
        ).stdout

        self.diff_full = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True,
            cwd=self.project_dir
        ).stdout

        return self.diff_stat, self.diff_full

    def run_tests(self, test_command=None):
        """Run the project's test suite."""
        cmd = test_command or ["python", "-m", "pytest", "tests/", "-q", "--tb=line"]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.project_dir, timeout=120
            )
            self.tests_passed = result.returncode == 0
            self.test_output = _strip_ansi(result.stdout + result.stderr)
            return self.tests_passed, self.test_output
        except subprocess.TimeoutExpired:
            self.tests_passed = False
            self.test_output = "[JARVIS] Tests timed out after 2 minutes"
            return False, self.test_output
        except FileNotFoundError:
            self.tests_passed = None
            self.test_output = "[JARVIS] Test runner not found"
            return None, self.test_output

    def commit(self, message):
        """Stage and commit all changes on the jarvis branch."""
        subprocess.run(["git", "add", "-A"],
                       cwd=self.project_dir, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.project_dir, capture_output=True, text=True
        )
        return result.returncode == 0

    def revert(self):
        """Discard all changes and delete the jarvis branch."""
        subprocess.run(["git", "checkout", "."],
                       cwd=self.project_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=self.project_dir, capture_output=True)
        if self.original_branch and self.branch_name:
            subprocess.run(
                ["git", "checkout", self.original_branch],
                cwd=self.project_dir, capture_output=True
            )
            subprocess.run(
                ["git", "branch", "-D", self.branch_name],
                cwd=self.project_dir, capture_output=True
            )
        self.branch_name = None
        return True

    def get_summary(self):
        """Build a human-readable summary of the pipeline run."""
        lines = []
        if self.branch_name:
            lines.append(f"Branch: {self.branch_name}")
        if self.diff_stat:
            lines.append(f"\nFiles changed:\n{self.diff_stat}")
        if self.tests_passed is True:
            lines.append("Tests: PASSED")
        elif self.tests_passed is False:
            lines.append("Tests: FAILED")
            if self.test_output:
                last_lines = self.test_output.strip().split('\n')[-5:]
                lines.append('\n'.join(last_lines))
        else:
            lines.append("Tests: not run")
        return '\n'.join(lines)
