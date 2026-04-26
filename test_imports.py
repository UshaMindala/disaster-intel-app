"""
Utility: write AWS credentials from .streamlit/secrets.toml into a local .env file.
Run this once when setting up a new dev environment.
Credentials are read from .streamlit/secrets.toml (gitignored) — never hardcoded here.
"""
import os
import tomllib
from pathlib import Path

SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"
ENV_FILE = Path(__file__).parent / ".env"

if not SECRETS_FILE.exists():
    print(f"ERROR: {SECRETS_FILE} not found. Copy secrets.toml.example and fill in values.")
    raise SystemExit(1)

with open(SECRETS_FILE, "rb") as f:
    secrets = tomllib.load(f)

lines = [f'{k}={v}' for k, v in secrets.items()]
ENV_FILE.write_text("\n".join(lines) + "\n")
print(f"Written {len(lines)} keys to {ENV_FILE}")

# Quick verification (no secret values printed)
from dotenv import load_dotenv
load_dotenv(ENV_FILE, override=True)
key = os.getenv("AWS_ACCESS_KEY_ID", "NOT SET")
print(f"AWS_ACCESS_KEY_ID set: {bool(key and key != 'NOT SET')}")
print("Done")
