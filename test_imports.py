"""
Write credentials from .streamlit/secrets.toml into files/.env
Run this each time you get new AWS credentials.
"""
from pathlib import Path

secrets_file = Path(__file__).parent / ".streamlit" / "secrets.toml"
env_file     = Path(__file__).parent / ".env"

if not secrets_file.exists():
    print(f"ERROR: {secrets_file} not found.")
    print("Edit .streamlit/secrets.toml with your credentials first.")
    raise SystemExit(1)

# Parse the TOML manually (avoid extra dependency)
lines = []
for raw in secrets_file.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    key, _, val = line.partition("=")
    lines.append(f"{key.strip()}={val.strip().strip(chr(34))}")

env_file.write_text("\n".join(lines) + "\n")
print("Written to:", env_file)

import os
from dotenv import load_dotenv
load_dotenv(env_file, override=True)
print("KEY:  ", (os.getenv("AWS_ACCESS_KEY_ID", "NOT SET") or "")[:20], "...")
print("Done — restart uvicorn to pick up new credentials")
