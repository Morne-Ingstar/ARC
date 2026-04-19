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

import re
import subprocess
from datetime import datetime
from pathlib import Path

# Strip ANSI terminal color codes from Claude CLI output
# (CLI outputs rich terminal formatting that renders as garbage in Tkinter)
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub('', text) if text else ''


class JarvisPipeline:

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.branch_name = None
        self.original_branch = None
        self.execution_log = None
        self.diff_stat = None
        self.diff_full = None
        self.test_output = None
        self.tests_passed = None

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

    def execute(self, prompt, budget_usd=1.0):
        """Run Claude Code CLI with the prompt.

        SECURITY: --allowed-tools "Read Edit" restricts Code to file
        reading and editing ONLY. No bash, no pip, no shell commands.
        Enforced at CLI level -- LLM cannot override.

        Returns:
            (success: bool, output: str) -- output is ANSI-stripped
        """
        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--print",
                    "--output-format", "text",
                    "--permission-mode", "auto",
                    "--allowed-tools", "Read Edit",
                    "--max-budget-usd", str(budget_usd),
                ],
                capture_output=True, text=True,
                cwd=str(self.project_dir),
                timeout=300
            )
            # Strip ANSI codes -- CLI uses terminal colors that
            # render as garbage in Tkinter text widgets
            clean_out = _strip_ansi(result.stdout)
            clean_err = _strip_ansi(result.stderr)
            self.execution_log = clean_out
            if result.returncode != 0:
                self.execution_log += f"\n\nSTDERR:\n{clean_err}"
            return result.returncode == 0, clean_out
        except subprocess.TimeoutExpired:
            self.execution_log = "[JARVIS] Claude Code timed out after 5 minutes"
            return False, self.execution_log
        except FileNotFoundError:
            self.execution_log = "[JARVIS] 'claude' CLI not found. Install Claude Code first."
            return False, self.execution_log

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
