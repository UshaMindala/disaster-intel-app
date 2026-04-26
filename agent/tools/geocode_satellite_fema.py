"""
Tool: Geocode — Street names → lat/lon via Nominatim
Tool: Satellite — Pre-fire image analysis via Pegasus
Tool: FEMA — Synthetic damage reports generation
"""
import json
import logging
import math
import random
import time
import uuid
import base64
from datetime import datetime, timedelta

import boto3
import httpx

from config import (
    AWS_REGION, AWS_ACCOUNT_ID, S3_BUCKET,
    S3_SATELLITE_PFX, S3_REPORTS_PFX, PEGASUS_MODEL_ID
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# GEOCODE
# ════════════════════════════════════════════════════════════════

def geocode_street(street_name: str, city: str = "Los Angeles, CA") -> dict:
    """Geocode a street name using Nominatim (free, no API key)."""
    query   = f"{street_name}, {city}"
    url     = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "disaster-intel-hackathon/1.0"}
    params  = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}

    try:
        r       = httpx.get(url, params=params, headers=headers, timeout=10)
        results = r.json()
        if results:
            return {
                "street_name":  street_name,
                "lat":          float(results[0]["lat"]),
                "lon":          float(results[0]["lon"]),
                "display_name": results[0]["display_name"],
                "geocoded":     True,
            }
    except Exception as e:
        logger.warning(f"Geocode failed {street_name}: {e}")

    return {"street_name": street_name, "lat": None, "lon": None, "geocoded": False}


def geocode_all_streets(streets: list, scenario: dict = None) -> tuple:
    """
    Geocode all streets. Returns (geocoded_streets, bbox, center).
    Uses GPS anchors for streets that fail Nominatim.
    """
    city = "Los Angeles, CA"
    if scenario:
        name = scenario.get("name", "")
        if "Fort Myers" in name or "Cape Coral" in name:
            city = "Fort Myers, FL"
        elif "Palisades" in name or "Los Angeles" in name:
            city = "Pacific Palisades, Los Angeles, CA"

    geocoded = []
    for i, s in enumerate(streets):
        result = geocode_street(s["street_name"], city)
        merged = {**s, **result}

        # Fallback: GPS anchor interpolation
        if not merged.get("lat") and scenario:
            t = _parse_timestamp(s.get("timestamp", "0:00"))
            lat, lon = _interpolate_gps(t, scenario.get("gps_anchors", []))
            merged["lat"]     = lat
            merged["lon"]     = lon
            merged["geocoded"] = False
            merged["geocode_method"] = "gps_interpolation"
        else:
            merged["geocode_method"] = "nominatim"

        geocoded.append(merged)
        time.sleep(1)  # Nominatim rate limit
        if (i + 1) % 10 == 0:
            logger.info(f"  Geocoded {i+1}/{len(streets)}")

    # Build bounding box
    valid = [s for s in geocoded if s.get("lat") and s.get("lon")]
    if not valid:
        bbox   = scenario["bbox"] if scenario else {}
        center = scenario["center"] if scenario else {}
    else:
        lats   = [s["lat"] for s in valid]
        lons   = [s["lon"] for s in valid]
        BUFFER = 0.005
        bbox   = {
            "min_lat": min(lats) - BUFFER, "max_lat": max(lats) + BUFFER,
            "min_lon": min(lons) - BUFFER, "max_lon": max(lons) + BUFFER,
        }
        center = {
            "lat": (bbox["min_lat"] + bbox["max_lat"]) / 2,
            "lon": (bbox["min_lon"] + bbox["max_lon"]) / 2,
        }

    found = sum(1 for s in geocoded if s.get("geocoded"))
    logger.info(f"✅ Geocoded: {found}/{len(streets)} via Nominatim")
    return geocoded, bbox, center


def _parse_timestamp(ts: str) -> float:
    """Convert mm:ss or ss string to float seconds."""
    try:
        parts = ts.replace(":", " ").split()
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return 0.0


def _interpolate_gps(t_s: float, anchors: list) -> tuple:
    """Linear GPS interpolation between anchor points."""
    if not anchors:
        return None, None
    anchors = sorted(anchors, key=lambda a: a["t"])
    if t_s <= anchors[0]["t"]:
        return anchors[0]["lat"], anchors[0]["lon"]
    if t_s >= anchors[-1]["t"]:
        return anchors[-1]["lat"], anchors[-1]["lon"]
    for i in range(len(anchors) - 1):
        a0, a1 = anchors[i], anchors[i+1]
        if a0["t"] <= t_s <= a1["t"]:
            r = (t_s - a0["t"]) / (a1["t"] - a0["t"])
            return (round(a0["lat"] + r*(a1["lat"]-a0["lat"]), 6),
                    round(a0["lon"] + r*(a1["lon"]-a0["lon"]), 6))
    return anchors[-1]["lat"], anchors[-1]["lon"]


# ════════════════════════════════════════════════════════════════
# SATELLITE — Pre-fire image analysis
# ════════════════════════════════════════════════════════════════

def upload_satellite_image(local_path: str) -> str:
    """Upload Google Earth Pro screenshot to S3."""
    import os
    session  = boto3.Session(region_name=AWS_REGION)
    s3       = session.client("s3")
    filename = os.path.basename(local_path)
    s3_key   = f"{S3_SATELLITE_PFX}/{filename}"
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    uri = f"s3://{S3_BUCKET}/{s3_key}"
    logger.info(f"✅ Satellite image uploaded: {uri}")
    return uri


def analyze_satellite_image(satellite_s3_uri: str) -> dict:
    """
    Send pre-fire satellite image to Pegasus for baseline analysis.
    Returns structure count, vegetation, road list, density.
    """
    session = boto3.Session(region_name=AWS_REGION)
    bedrock = session.client("bedrock-runtime")

    prompt = """
This is a pre-disaster satellite image taken BEFORE the disaster event.
Analyze this image and provide a baseline assessment:

1. TOTAL STRUCTURES VISIBLE — estimated count of all buildings/homes
2. VEGETATION COVERAGE — percentage of area covered by trees/vegetation
3. ROAD NAMES VISIBLE — list all road or street names you can read
4. NEIGHBORHOOD DENSITY — sparse / medium / dense residential
5. LAND USE — describe the types of land use visible
   (residential, commercial, industrial, open space, mixed)
6. NOTABLE LANDMARKS — schools, parks, commercial centers, bridges
7. INFRASTRUCTURE — describe visible roads, utilities, water features
8. BASELINE DESCRIPTION — overall description of the pre-disaster state
"""
    request_body = {
        "inputPrompt": prompt,
        "mediaSource": {"s3Location": {
            "uri":         satellite_s3_uri,
            "bucketOwner": AWS_ACCOUNT_ID,
        }},
        "temperature": 0,
        "responseFormat": {"jsonSchema": {
            "type": "object",
            "properties": {
                "total_structures":          {"type": "integer"},
                "vegetation_coverage_pct":   {"type": "number"},
                "roads_visible":             {"type": "array", "items": {"type": "string"}},
                "density":                   {"type": "string",
                                              "enum": ["sparse","medium","dense"]},
                "land_use":                  {"type": "array", "items": {"type": "string"}},
                "landmarks":                 {"type": "array", "items": {"type": "string"}},
                "infrastructure_description":{"type": "string"},
                "baseline_description":      {"type": "string"},
            },
            "required": ["total_structures", "baseline_description"]
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
        result = json.loads(body["message"])
        logger.info(f"✅ Satellite baseline: {result.get('total_structures')} structures")
        return result
    except Exception as e:
        logger.warning(f"Satellite parse failed: {e}")
        return {
            "total_structures":        500,
            "vegetation_coverage_pct": 40,
            "roads_visible":           [],
            "density":                 "dense",
            "baseline_description":    body.get("message", "")[:300],
        }


# ════════════════════════════════════════════════════════════════
# FEMA REPORTS — Synthetic generation
# ════════════════════════════════════════════════════════════════

SEV_ORDER = ["none","minor","moderate","severe","destroyed"]

LOCATION_TEMPLATES = {
    "hurricane": [
        ("Fort Myers Beach marina district",    26.4520, -81.9526),
        ("Estero Island beachfront",            26.4489, -81.9489),
        ("San Carlos Blvd intersection",        26.4510, -81.9510),
        ("Del Prado Blvd commercial corridor",  26.6389, -81.8701),
        ("Cape Coral retail strip US-41",       26.6406, -81.8723),
    ],
    "wildfire": [
        ("Galloway St residential area",        34.0441, -118.5268),
        ("Temescal Canyon Rd corridor",         34.0448, -118.5260),
        ("Via de la Paz properties",            34.0461, -118.5275),
        ("Albright St neighborhood",            34.0445, -118.5255),
        ("Fiske St residential block",          34.0435, -118.5248),
    ],
    "generic": [
        ("Main St downtown area",               0.0, 0.0),
        ("Highway 1 corridor",                  0.0, 0.0),
        ("Industrial district",                 0.0, 0.0),
    ],
}


def generate_fema_reports(scenario: dict = None, count: int = 4) -> tuple:
    """Generate synthetic FEMA reports with intentional severity bias."""
    disaster_type = scenario.get("type", "generic") if scenario else "generic"
    locations     = LOCATION_TEMPLATES.get(disaster_type,
                                           LOCATION_TEMPLATES["generic"])
    incident_name = scenario.get("name", "Disaster Event") if scenario else "Disaster"
    base_date     = datetime.strptime(
        scenario["date"] if scenario else "2024-01-01", "%Y-%m-%d"
    )

    configs = [
        {"file":"fema_day1_initial.json",    "date_offset":1, "bias":"under", "n":8,
         "title":"Initial Rapid Assessment — Day 1"},
        {"file":"fema_day3_residential.json","date_offset":3, "bias":"exact", "n":12,
         "title":"Residential Survey — Day 3"},
        {"file":"fema_day5_infrastructure.json","date_offset":5,"bias":"exact","n":10,
         "title":"Infrastructure Assessment — Day 5"},
        {"file":"fema_day7_commercial.json", "date_offset":7, "bias":"over",  "n":9,
         "title":"Commercial District Summary — Day 7"},
    ]

    reports    = []
    all_claims = []

    for cfg in configs[:count]:
        report_id   = str(uuid.uuid4())
        report_date = (base_date + timedelta(days=cfg["date_offset"])).strftime("%Y-%m-%d")
        claims      = _make_claims(report_id, cfg["n"], cfg["bias"],
                                   report_date, locations, incident_name)
        reports.append({
            "report_id":    report_id,
            "file":         cfg["file"],
            "title":        cfg["title"],
            "date":         report_date,
            "incident":     incident_name,
            "bias":         cfg["bias"],
            "claims":       claims,
        })
        all_claims.extend(claims)

    logger.info(f"✅ FEMA reports: {len(reports)} | {len(all_claims)} claims")
    return reports, all_claims


def _make_claims(report_id, n, bias, date, locations, incident):
    claims = []
    for i in range(n):
        loc      = random.choice(locations)
        true_sev = random.choice(["minor","moderate","severe","destroyed"])
        sev      = (SEV_ORDER[max(0, SEV_ORDER.index(true_sev)-1)] if bias=="under" else
                    SEV_ORDER[min(4, SEV_ORDER.index(true_sev)+1)] if bias=="over" else
                    true_sev)
        infra = ("road" if "blvd" in loc[0].lower() or "hwy" in loc[0].lower()
                 else "commercial" if "commercial" in loc[0].lower()
                 else "residential")
        desc  = f"{sev.upper()} {infra} damage at {loc[0]}. Field report {date}."
        # Simulate 15% OCR errors
        if random.random() < 0.15:
            desc = desc.replace("road","r0ad").replace("roof","r00f") + " [OCR:LOW]"

        claims.append({
            "claim_id":    str(uuid.uuid4())[:8],
            "report_id":   report_id,
            "location":    loc[0],
            "lat":         loc[1] + random.uniform(-0.003, 0.003),
            "lon":         loc[2] + random.uniform(-0.003, 0.003),
            "severity":    sev,
            "infrastructure": infra,
            "description": desc,
            "date":        date,
            "page":        i // 3 + 1,
            "has_ocr_error": "OCR:LOW" in desc,
        })
    return claims
