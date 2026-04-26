"""
Tool: Pegasus 1.2 — Video-to-Text Damage Analysis
Handles: full video summary + per-street structured extraction
"""
import json
import logging
import re

import boto3
from config import AWS_REGION, AWS_ACCOUNT_ID, S3_BUCKET, PEGASUS_MODEL_ID

logger = logging.getLogger(__name__)


def _bedrock():
    return boto3.Session(region_name=AWS_REGION).client("bedrock-runtime")


# ── Full video streaming summary ──────────────────────────────

def full_video_summary(video_s3_uri: str) -> str:
    """Streaming Pegasus analysis of entire video."""
    bedrock = _bedrock()
    prompt = """
You are a FEMA damage assessment analyst reviewing aerial disaster footage.
Provide a comprehensive assessment covering:
1. SCENE OVERVIEW: Describe what disaster type and geographic areas are visible
2. DAMAGE INVENTORY: All damage types observed with approximate timestamps
3. SEVERITY DISTRIBUTION: Estimated % of structures with minor/moderate/severe/destroyed damage
4. INFRASTRUCTURE STATUS: Roads, bridges, utilities — passable vs blocked
5. PRIORITY ZONES: Top 3 highest-priority areas for emergency response and why
6. HAZARDS: Visible safety hazards responders should know about
7. GAPS: What cannot be determined from video alone
"""
    request_body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "s3Location": {
                "uri":         video_s3_uri,
                "bucketOwner": AWS_ACCOUNT_ID,
            }
        },
        "temperature": 0,
    }

    try:
        streaming = bedrock.invoke_model_with_response_stream(
            modelId=PEGASUS_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        summary = ""
        for event in streaming["body"]:
            chunk    = json.loads(event["chunk"]["bytes"])
            txt      = chunk.get("message", "")
            summary += txt
        logger.info(f"✅ Full video summary: {len(summary)} chars")
        return summary
    except Exception as e:
        logger.warning(f"Pegasus full summary failed: {e}")
        return f"Video analysis unavailable: {e}"

# ── Street-level structured extraction ───────────────────────

STREET_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "street_name":          {"type": "string"},
            "timestamp":            {"type": "string"},
            "burn_status":          {"type": "string", "enum": [
                                     "completely_destroyed", "mostly_destroyed",
                                     "partially_burned", "minor_damage",
                                     "intact", "unknown"]},
            "flood_status":         {"type": "string", "enum": [
                                     "completely_submerged", "partially_flooded",
                                     "debris_blocked", "accessible", "unknown"]},
            "damage_type":          {"type": "string", "enum": [
                                     "fire", "flood", "wind", "structural",
                                     "debris", "combined", "unknown"]},
            "structures_destroyed": {"type": "integer"},
            "structures_intact":    {"type": "integer"},
            "emergency_vehicles":   {"type": "boolean"},
            "accessibility":        {"type": "string", "enum": [
                                     "passable", "debris_blocked",
                                     "boat_only", "inaccessible", "unknown"]},
            "hazards":              {"type": "array", "items": {"type": "string"}},
            "notable_observations": {"type": "string"},
            "priority_score":       {"type": "integer",
                                     "minimum": 1, "maximum": 10},
        },
        "required": ["street_name", "damage_type", "structures_destroyed",
                     "structures_intact", "accessibility", "priority_score"]
    }
}

STREET_PROMPT = """
You are a FEMA damage assessment analyst reviewing aerial disaster footage.

The video may show street name labels overlaid on the footage, or street names
may be visible on signage. Identify EVERY street, road, boulevard, or named 
thoroughfare visible in this video.

For EVERY street found provide:
1. STREET NAME — exact name as visible
2. TIMESTAMP — when this street appears (mm:ss)
3. BURN STATUS — for wildfires: completely_destroyed / mostly_destroyed / 
   partially_burned / minor_damage / intact / unknown
4. FLOOD STATUS — for floods: completely_submerged / partially_flooded / 
   debris_blocked / accessible / unknown
5. DAMAGE TYPE — fire / flood / wind / structural / debris / combined / unknown
6. STRUCTURES DESTROYED — estimated count
7. STRUCTURES INTACT — estimated count
8. EMERGENCY VEHICLES — true/false
9. ACCESSIBILITY — passable / debris_blocked / boat_only / inaccessible / unknown
10. HAZARDS — list any visible hazards
11. NOTABLE OBSERVATIONS — standing chimneys, burned vehicles, active hotspots etc.
12. PRIORITY SCORE — 1 (lowest) to 10 (highest) for emergency response urgency

Do not skip any street. Include every road visible even if only briefly shown.
"""


def extract_streets_structured(video_s3_uri: str) -> tuple:
    """
    Pegasus structured extraction with JSON schema.
    Returns (streets_list, raw_output).
    Falls back to text parsing if JSON fails.
    """
    bedrock = _bedrock()
    request_body = {
        "inputPrompt": STREET_PROMPT,
        "mediaSource": {"s3Location": {
            "uri":         video_s3_uri,
            "bucketOwner": AWS_ACCOUNT_ID,
        }},
        "temperature": 0,
        "responseFormat": {"jsonSchema": STREET_SCHEMA},
    }
    resp     = bedrock.invoke_model(
        modelId=PEGASUS_MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )
    body      = json.loads(resp["body"].read())
    raw       = body.get("message", "")

    try:
        streets = json.loads(raw)
        if isinstance(streets, list):
            logger.info(f"✅ Structured extraction: {len(streets)} streets")
            return streets, raw
    except json.JSONDecodeError:
        pass

    # Fallback: parse markdown text
    logger.warning("JSON parse failed — falling back to text parser")
    streets = parse_streets_from_text(raw)
    return streets, raw


def parse_streets_from_text(raw_text: str) -> list:
    """
    Parse Pegasus markdown-style text output into street dicts.
    Handles the formatted output Pegasus returns when schema fails.
    """
    streets = []
    entries = re.split(r'\n(?=\d+\.\s+\*\*STREET|\d+\.\s+Street)', raw_text)

    for entry in entries:
        if not entry.strip():
            continue
        s = {}

        # Street name
        name = re.search(r'(?:STREET NAME|Street Name).*?:\*\*\s*(.+)|^\d+\.\s+(.+?)(?:\n|$)', entry)
        if name:
            s["street_name"] = (name.group(1) or name.group(2) or "").strip()
        if not s.get("street_name"):
            continue

        # Timestamp
        ts = re.search(r'TIMESTAMP.*?:\*\*\s*(.+)', entry)
        s["timestamp"] = ts.group(1).strip() if ts else "unknown"

        # Burn status
        burn = re.search(r'BURN STATUS.*?:\*\*\s*(.+)', entry)
        burn_raw = burn.group(1).lower() if burn else ""
        s["burn_status"] = (
            "completely_destroyed" if "completely" in burn_raw else
            "mostly_destroyed"     if "mostly"     in burn_raw else
            "partially_burned"     if "partially"  in burn_raw else
            "minor_damage"         if "minor"      in burn_raw else
            "intact"               if "intact"     in burn_raw else
            "unknown"
        )

        # Damage type
        dmg = re.search(r'DAMAGE TYPE.*?:\*\*\s*(.+)', entry)
        dmg_raw = dmg.group(1).lower() if dmg else ""
        s["damage_type"] = (
            "fire"       if "fire"       in dmg_raw else
            "flood"      if "flood"      in dmg_raw else
            "wind"       if "wind"       in dmg_raw else
            "structural" if "structural" in dmg_raw else
            "combined"   if "combined"   in dmg_raw else
            "unknown"
        )

        # Structures
        dest   = re.search(r'STRUCTURES DESTROYED.*?:\*\*\s*(\d+)', entry)
        intact = re.search(r'STRUCTURES INTACT.*?:\*\*\s*(\d+)', entry)
        s["structures_destroyed"] = int(dest.group(1))   if dest   else 0
        s["structures_intact"]    = int(intact.group(1)) if intact else 0

        # Emergency vehicles
        ev     = re.search(r'EMERGENCY VEHICLES.*?:\*\*\s*(.+)', entry)
        ev_raw = ev.group(1).lower() if ev else ""
        s["emergency_vehicles"] = "yes" in ev_raw or "true" in ev_raw

        # Accessibility
        acc     = re.search(r'ACCESSIBILITY.*?:\*\*\s*(.+)', entry)
        acc_raw = acc.group(1).lower() if acc else ""
        s["accessibility"] = (
            "passable"      if "passable"   in acc_raw else
            "debris_blocked"if "debris"     in acc_raw else
            "boat_only"     if "boat"       in acc_raw else
            "inaccessible"  if "inaccessible" in acc_raw else
            "unknown"
        )

        # Hazards
        haz    = re.search(r'HAZARDS.*?:\*\*\s*(.+)', entry)
        s["hazards"] = [h.strip() for h in haz.group(1).split(",")] if haz else []

        # Notable observations
        obs    = re.search(r'NOTABLE OBSERVATIONS.*?:\*\*\s*(.+)', entry)
        s["notable_observations"] = obs.group(1).strip() if obs else ""

        # Priority score
        pri    = re.search(r'PRIORITY.*?:\*\*\s*(\d+)', entry)
        s["priority_score"] = int(pri.group(1)) if pri else {
            "completely_destroyed": 10, "mostly_destroyed": 8,
            "partially_burned": 6,      "minor_damage": 3,
            "intact": 1,                "unknown": 5,
        }.get(s["burn_status"], 5)

        streets.append(s)

    logger.info(f"✅ Text parsed: {len(streets)} streets")
    return streets


# ── Emergency response prompt ─────────────────────────────────

def emergency_assessment(video_s3_uri: str) -> dict:
    """Get emergency response priorities from Pegasus."""
    bedrock = _bedrock()
    prompt  = """
You are a FEMA search and rescue coordinator.
Based on this aerial disaster footage provide:
1. The single highest priority location (street + cross street + reason)
2. Top 5 priority locations ranked
3. Accessible entry routes for emergency vehicles
4. Blocked or dangerous routes
5. Any survivor indicators visible
6. Structural hazards for responders
7. Best staging area location
8. No-go zones
"""
    request_body = {
        "inputPrompt": prompt,
        "mediaSource": {"s3Location": {
            "uri":         video_s3_uri,
            "bucketOwner": AWS_ACCOUNT_ID,
        }},
        "temperature": 0,
        "responseFormat": {"jsonSchema": {
            "type": "object",
            "properties": {
                "highest_priority":  {"type": "object", "properties": {
                    "location":     {"type": "string"},
                    "reason":       {"type": "string"},
                    "support_type": {"type": "string"},
                }},
                "top_5_priorities":  {"type": "array", "items": {
                    "type": "object", "properties": {
                        "rank":         {"type": "integer"},
                        "location":     {"type": "string"},
                        "reason":       {"type": "string"},
                        "support_type": {"type": "string"},
                    }
                }},
                "accessible_routes": {"type": "array", "items": {"type": "string"}},
                "blocked_routes":    {"type": "array", "items": {"type": "string"}},
                "survivor_indicators": {"type": "string"},
                "structural_hazards":  {"type": "array", "items": {"type": "string"}},
                "staging_area":        {"type": "string"},
                "no_go_zones":         {"type": "array", "items": {"type": "string"}},
            },
            "required": ["highest_priority", "top_5_priorities",
                         "accessible_routes", "blocked_routes"]
        }},
    }
    resp = bedrock.invoke_model(
        modelId=PEGASUS_MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )
    body = json.loads(resp["body"].read())
    try:
        return json.loads(body["message"])
    except Exception:
        return {"raw": body.get("message", "")}


# ── Natural language query ─────────────────────────────────────

def nl_query_video(video_s3_uri: str, question: str) -> str:
    """Answer any natural language question about the video."""
    bedrock = _bedrock()
    request_body = {
        "inputPrompt": question,
        "mediaSource": {"s3Location": {
            "uri":         video_s3_uri,
            "bucketOwner": AWS_ACCOUNT_ID,
        }},
        "temperature": 0,
    }
    streaming = bedrock.invoke_model_with_response_stream(
        modelId=PEGASUS_MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )
    answer = ""
    for event in streaming["body"]:
        chunk   = json.loads(event["chunk"]["bytes"])
        answer += chunk.get("message", "")
    return answer
