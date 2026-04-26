"""
Disaster Intelligence System — Configuration
All settings loaded from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this file (files/.env).
# override=True ensures this wins over any stale shell exports.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── AWS ──────────────────────────────────────────────────────
AWS_REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip('"').strip("'")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "371542051390").strip('"').strip("'")

S3_BUCKET      = os.getenv("S3_BUCKET_NAME",
                    "twelvelabs-bedrock-workshop-workshopbucket-yaqtzrqyuzku").strip('"').strip("'")
S3_VECTOR_BUCKET = os.getenv("S3_VECTOR_BUCKET_NAME",
                    "twelvelabs-aws-vectorbucket-kfzwf0z86w95").strip('"').strip("'")

# S3 path prefixes
S3_VIDEOS_PFX      = "videos"
S3_EMBEDDINGS_PFX  = "embeddings"
S3_CLIPS_PFX       = "clips"
S3_OUTPUTS_PFX     = "outputs"
S3_REPORTS_PFX     = "reports"
S3_SATELLITE_PFX   = "satellite"
S3_VECTOR_INDEX    = "damage-events-index"

# ── TwelveLabs Models ────────────────────────────────────────
MARENGO_MODEL_ID = "us.twelvelabs.marengo-embed-3-0-v1:0"
PEGASUS_MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"

# ── Overture Maps ─────────────────────────────────────────────
OVERTURE_BUCKET  = "overturemaps-us-west-2"
OVERTURE_RELEASE = "release/2024-09-18.0"

# ── Fusion Engine ─────────────────────────────────────────────
SPATIAL_THRESHOLD_M    = 300   # entity resolution radius
TEMPORAL_WINDOW_HOURS  = 48    # report vs video matching
MARENGO_MIN_SCORE      = 0.45  # similarity threshold
MARENGO_TOP_K          = 5     # results per query
MAX_COMPARISON_PAIRS   = 50000 # complexity guard

# ── Known Video Scenarios ─────────────────────────────────────
VIDEO_SCENARIOS = {
    "zJyDF8_NHcs.mp4": {
        "name":     "Hurricane Ian — Fort Myers Beach / Cape Coral, FL",
        "type":     "hurricane",
        "date":     "2022-09-28",
        "center":   {"lat": 26.55, "lon": -81.91},
        "bbox":     {"min_lon": -81.98, "min_lat": 26.42,
                     "max_lon": -81.87, "max_lat": 26.68},
        "gps_anchors": [
            {"t": 10,  "lat": 26.4520, "lon": -81.9526},
            {"t": 90,  "lat": 26.4489, "lon": -81.9489},
            {"t": 180, "lat": 26.6406, "lon": -81.8723},
            {"t": 300, "lat": 26.6389, "lon": -81.8701},
        ],
        "damage_queries": [
            "collapsed or destroyed building",
            "debris scattered on street",
            "missing roof structure",
            "damaged marina or dock",
            "flooded road or intersection",
            "submerged vehicle",
            "flooded commercial building",
            "impassable road water",
            "water covering parking lot",
            "structural damage residential building",
        ],
    },
    "Palisades_Wildfire.mp4": {
        "name":     "Palisades Fire — Pacific Palisades, LA",
        "type":     "wildfire",
        "date":     "2025-01-07",
        "center":   {"lat": 34.0450, "lon": -118.5260},
        "bbox":     {"min_lon": -118.56, "min_lat": 34.03,
                     "max_lon": -118.49, "max_lat": 34.06},
        "gps_anchors": [
            {"t": 5,   "lat": 34.0441, "lon": -118.5268},
            {"t": 120, "lat": 34.0445, "lon": -118.5255},
            {"t": 240, "lat": 34.0461, "lon": -118.5275},
            {"t": 360, "lat": 34.0448, "lon": -118.5260},
            {"t": 480, "lat": 34.0435, "lon": -118.5248},
        ],
        "damage_queries": [
            "completely destroyed house fire damage",
            "burned building rubble ash",
            "charred remains residential structure",
            "fire damaged neighborhood aerial view",
            "destroyed roof foundation only remaining",
            "burned vehicles in driveway",
            "emergency vehicles on burned street",
            "intact structure surrounded by destruction",
            "fire perimeter boundary burned vs unburned",
            "chimney standing destroyed home",
        ],
    },
}

# Generic queries for unknown video types
GENERIC_DAMAGE_QUERIES = [
    "destroyed or collapsed structure",
    "severe infrastructure damage",
    "emergency vehicles disaster response",
    "debris blocked road",
    "flooded or burned area",
    "damaged residential building",
    "impassable road or bridge",
    "displaced vehicles or objects",
    "utility infrastructure damage",
    "disaster affected neighborhood",
]

# ── App ────────────────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 8000
DEBUG    = os.getenv("DEBUG", "false").lower() == "true"

# ── Severity scoring ───────────────────────────────────────────
SEVERITY_SCORES = {
    "none": 0, "minor": 25, "moderate": 50,
    "severe": 75, "destroyed": 100,
    "completely_destroyed": 100, "mostly_destroyed": 80,
    "partially_burned": 55, "minor_damage": 25,
    "intact": 0, "unknown": 40,
}
