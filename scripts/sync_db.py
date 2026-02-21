"""
DB sync helper — pull the latest CI database from GitHub Actions to local,
or push the local DB up by triggering a workflow dispatch.

Usage:
    python scripts/sync_db.py pull    # download latest CI DB to local
    python scripts/sync_db.py push    # trigger CI run to re-seed from local (via workflow_dispatch)
    python scripts/sync_db.py status  # show local vs CI pick counts

Requirements:
    pip install requests
    Set env var GITHUB_TOKEN with a Personal Access Token (repo + actions scope).
    Get one at: https://github.com/settings/tokens
"""

import os
import sys
import shutil
import sqlite3
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("Install requests first:  pip install requests")
    sys.exit(1)

REPO = "NikolayAngelov90/betting-agent"
ARTIFACT_NAME = "betting-db-latest"
LOCAL_DB = Path("data/football_betting.db")
API_BASE = f"https://api.github.com/repos/{REPO}"


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("ERROR: Set the GITHUB_TOKEN environment variable.")
        print("  Get a token at: https://github.com/settings/tokens")
        print("  Scopes needed: repo, actions:read")
        sys.exit(1)
    return token


def headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def db_summary(path: Path) -> str:
    if not path.exists():
        return "  (file not found)"
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM saved_picks")
    picks = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM saved_picks WHERE result IS NULL")
    pending = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM matches WHERE is_fixture=0")
    results = cur.fetchone()[0]
    cur.execute("SELECT MAX(pick_date) FROM saved_picks")
    latest = cur.fetchone()[0] or "none"
    conn.close()
    return f"  {picks} picks ({pending} pending) | {results} match results | latest pick: {latest}"


def cmd_status(token: str):
    print("=== LOCAL DB ===")
    print(db_summary(LOCAL_DB))

    print("\n=== GITHUB CI DB (latest artifact) ===")
    r = requests.get(f"{API_BASE}/actions/artifacts?name={ARTIFACT_NAME}&per_page=1",
                     headers=headers(token))
    r.raise_for_status()
    artifacts = r.json().get("artifacts", [])
    if not artifacts:
        print("  No artifact found yet (CI hasn't run since artifact upload was added).")
        return
    art = artifacts[0]
    created = art["created_at"]
    size_kb = art["size_in_bytes"] // 1024
    print(f"  Artifact created: {created}  ({size_kb} KB)")
    print("  Run 'pull' to download and inspect it.")


def cmd_pull(token: str):
    print("Fetching latest artifact list...")
    r = requests.get(f"{API_BASE}/actions/artifacts?name={ARTIFACT_NAME}&per_page=1",
                     headers=headers(token))
    r.raise_for_status()
    artifacts = r.json().get("artifacts", [])
    if not artifacts:
        print("No artifact found. The CI must run at least once after the workflow was updated.")
        sys.exit(1)

    art = artifacts[0]
    art_id = art["id"]
    created = art["created_at"]
    print(f"Found artifact #{art_id} created {created}")

    print("Downloading...")
    dl = requests.get(f"{API_BASE}/actions/artifacts/{art_id}/zip",
                      headers=headers(token), stream=True)
    dl.raise_for_status()

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "db.zip"
        with open(zip_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        extracted_db = Path(tmp) / "football_betting.db"
        if not extracted_db.exists():
            print("ERROR: DB file not found inside artifact zip.")
            sys.exit(1)

        # Backup existing local DB
        if LOCAL_DB.exists():
            backup = LOCAL_DB.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            shutil.copy2(LOCAL_DB, backup)
            print(f"Local DB backed up to: {backup}")

        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted_db, LOCAL_DB)

    print("\nSync complete!")
    print("=== PULLED DB ===")
    print(db_summary(LOCAL_DB))


def cmd_push(token: str):
    """Trigger a workflow_dispatch to run the CI pipeline immediately."""
    print("Triggering workflow_dispatch on main branch...")
    r = requests.post(
        f"{API_BASE}/actions/workflows/daily-picks.yml/dispatches",
        headers=headers(token),
        json={"ref": "main"},
    )
    if r.status_code == 204:
        print("CI run triggered successfully.")
        print(f"Check progress at: https://github.com/{REPO}/actions")
    else:
        print(f"ERROR {r.status_code}: {r.text}")
        sys.exit(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    tok = get_token()

    if cmd == "pull":
        cmd_pull(tok)
    elif cmd == "push":
        cmd_push(tok)
    elif cmd == "status":
        cmd_status(tok)
    else:
        print(__doc__)
