#!/usr/bin/env python3
"""
Simple Flask API server for the Codex Dashboard.
Provides endpoints to execute Python scripts directly from the web interface.
"""
import subprocess
import sys
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the dashboard

ROOT = Path(__file__).resolve().parent


def run_command(cmd: str) -> dict:
    """Run a shell command and return the result as a dict."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=300,  # 5 minute timeout
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out after 5 minutes",
        }
    except Exception as e:
        return {"success": False, "returncode": -1, "stdout": "", "stderr": str(e)}


@app.route("/run/ingest_repo", methods=["POST"])
def run_ingest_repo():
    """Execute the repo ingestion script."""
    cmd = f"{sys.executable} scripts/ingest_repo.py"
    result = run_command(cmd)
    return jsonify(result)


@app.route("/run/codex_plan", methods=["POST"])
def run_codex_plan():
    """Execute the codex planner."""
    cmd = f"{sys.executable} scripts/codex.py plan"
    result = run_command(cmd)
    return jsonify(result)


@app.route("/run/codex_execute", methods=["POST"])
def run_codex_execute():
    """Execute the codex executor."""
    cmd = f"{sys.executable} scripts/codex.py execute"
    result = run_command(cmd)
    return jsonify(result)


@app.route("/run/approval/<task_id>/<int:approved>", methods=["POST"])
def run_approval(task_id: str, approved: int):
    """Execute the approval command for a task."""
    # Validate task_id to prevent command injection
    if not task_id.replace("_", "").replace("-", "").isalnum():
        return jsonify(
            {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Invalid task ID",
            }
        )

    bool_literal = "True" if approved else "False"
    cmd = f"{sys.executable} -c \"import json, pathlib; p = pathlib.Path(r'.codex/approvals/tasks/{task_id}.approved.json'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps({{'approved': {bool_literal}}}, indent=2))\""
    result = run_command(cmd)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/.codex/<path:filename>")
def codex_files(filename):
    """Serve files from the .codex directory."""
    return send_from_directory(ROOT / ".codex", filename)


@app.route("/", methods=["GET"])
def dashboard():
    """Serve the dashboard HTML file."""
    html_path = ROOT / "scripts" / "codex_dashboard.html"
    print(f"Dashboard route called. Looking for file at: {html_path}")
    print(f"File exists: {html_path.exists()}")
    if html_path.exists():
        print("Returning HTML content")
        return html_path.read_text()
    else:
        print("File not found")
        return (
            f"Dashboard file not found at {html_path}. Please ensure scripts/codex_dashboard.html exists.",
            404,
        )


if __name__ == "__main__":
    print("Starting Codex Dashboard API server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=5000, debug=False)
