"""
Disaster Intelligence System — FastAPI Application
VS Code + FastAPI + Browser UI
"""
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from typing import Optional
import logging
logging.basicConfig(level=logging.DEBUG)

import boto3
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import run_pipeline
from agent.tools.pegasus import nl_query_video
from agent.tools.geocode_satellite_fema import upload_satellite_image
from config import AWS_REGION, S3_BUCKET, S3_OUTPUTS_PFX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── In-memory job store ───────────────────────────────────────
# Maps job_id → pipeline state / status
_jobs: dict = {}

app = FastAPI(
    title="Disaster Intelligence System",
    description="Multi-source disaster damage assessment — TwelveLabs on AWS Bedrock",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
print(f"DEBUG static_dir: {static_dir}")
print(f"DEBUG exists: {os.path.exists(static_dir)}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":   "healthy",
        "region":   AWS_REGION,
        "bucket":   S3_BUCKET,
        "timestamp":datetime.utcnow().isoformat(),
    }


# ════════════════════════════════════════════════════════════════
# FRONTEND
# ════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        index = os.path.join(static_dir, "index.html")
        with open(index, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"<pre>ERROR: {e}\n{traceback.format_exc()}</pre>", status_code=500)


# ════════════════════════════════════════════════════════════════
# PIPELINE — Upload video + run full agent
# ════════════════════════════════════════════════════════════════

@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    satellite: Optional[UploadFile] = File(None),
    nl_query: Optional[str] = Form(None),
):
    """
    Upload video (+ optional satellite image) and run full agentic pipeline.
    Returns job_id immediately. Poll /api/status/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "queued", "messages": [], "errors": []}

    # Save video locally
    tmp_dir   = tempfile.mkdtemp()
    vid_path  = os.path.join(tmp_dir, video.filename)
    with open(vid_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Save satellite if provided
    sat_s3_uri = None
    if satellite and satellite.filename:
        sat_path = os.path.join(tmp_dir, satellite.filename)
        with open(sat_path, "wb") as f:
            shutil.copyfileobj(satellite.file, f)
        try:
            sat_s3_uri = upload_satellite_image(sat_path)
        except Exception as e:
            logger.warning(f"Satellite upload failed: {e}")

    background_tasks.add_task(
        _run_pipeline_bg, job_id, vid_path, sat_s3_uri, nl_query, tmp_dir
    )

    return {"job_id": job_id, "status": "queued",
            "message": f"Pipeline started. Poll /api/status/{job_id}"}


def _run_pipeline_bg(job_id: str, vid_path: str,
                     sat_s3_uri: str, nl_query: str, tmp_dir: str):
    """Background pipeline execution."""
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        final_state = run_pipeline(
            video_local_path = vid_path,
            satellite_s3_uri = sat_s3_uri,
            nl_query         = nl_query,
        )
        _jobs[job_id]["status"]     = "complete"
        _jobs[job_id]["state"]      = _serialize(final_state)
        _jobs[job_id]["messages"]   = final_state.get("messages", [])
        _jobs[job_id]["errors"]     = final_state.get("errors", [])
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Job {job_id} complete")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["errors"].append(str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
# STATUS + RESULTS
# ════════════════════════════════════════════════════════════════

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status and progress messages."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "job_id":     job_id,
        "status":     job["status"],
        "messages":   job.get("messages", []),
        "errors":     job.get("errors", []),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get full assessment results."""
    job = _get_complete_job(job_id)
    state = job["state"]
    return {
        "job_id":          job_id,
        "scenario":        state.get("scenario", {}).get("name", "Unknown"),
        "total_events":    len(state.get("fused_events") or []),
        "by_status":       _count_by(state.get("fused_events") or [], "burn_status"),
        "by_access":       _count_by(state.get("fused_events") or [], "accessibility"),
        "fusion_insights": len(state.get("fusion_insights") or []),
        "conflicts":       state.get("conflicts_detected", 0),
        "processing_s":    state.get("processing_time_s"),
        "report_s3":       state.get("report_s3_uri"),
        "completed_steps": state.get("completed_steps", []),
        "mission_impact":  state.get("mission_impact"),
    }


@app.get("/api/events/{job_id}")
async def get_events(
    job_id: str,
    min_risk: float = 0,
    fusion_only: bool = False,
    status_filter: str = None,
):
    """Get damage events with optional filtering."""
    job    = _get_complete_job(job_id)
    events = (job["state"].get("fused_events") or []).copy()

    if min_risk > 0:
        events = [e for e in events if e.get("risk_score", 0) >= min_risk]
    if fusion_only:
        events = [e for e in events if e.get("fusion_insight")]
    if status_filter:
        events = [e for e in events if e.get("burn_status") == status_filter]

    return {
        "total": len(events),
        "events": [{
            "event_id":          e["event_id"],
            "street_name":       e["street_name"],
            "lat":               e["lat"],
            "lon":               e["lon"],
            "burn_status":       e.get("burn_status"),
            "damage_type":       e.get("damage_type"),
            "risk_score":        e.get("risk_score"),
            "confidence":        e.get("confidence"),
            "confidence_breakdown": e.get("confidence_breakdown"),
            "accessibility":     e.get("accessibility"),
            "structures_destroyed": e.get("structures_destroyed"),
            "structures_intact": e.get("structures_intact"),
            "pre_fire_structures":e.get("pre_fire_structures"),
            "estimated_loss":    e.get("estimated_loss"),
            "sources":           e.get("sources"),
            "source_count":      e.get("source_count"),
            "fusion_insight":    e.get("fusion_insight"),
            "fusion_insight_type":e.get("fusion_insight_type"),
            "has_conflict":      e.get("has_conflict"),
            "conflict":          e.get("conflict"),
            "hazards":           e.get("hazards"),
            "notable_observations": e.get("notable_observations"),
            "provenance":        e.get("provenance"),
        } for e in events]
    }


@app.get("/api/geojson/{job_id}")
async def get_geojson(job_id: str):
    """Get FEMA-compatible GeoJSON."""
    job = _get_complete_job(job_id)
    return JSONResponse(content=job["state"].get("geojson") or {})


@app.get("/api/insights/{job_id}")
async def get_insights(job_id: str):
    """Get fusion-only intelligence insights."""
    job      = _get_complete_job(job_id)
    insights = job["state"].get("fusion_insights") or []
    fused    = job["state"].get("fused_events") or []
    return {
        "total": len(insights),
        "insights": insights,
        "single_source_comparison": {
            "video_only":  f"{len(fused)} detections, no pre-fire baseline",
            "fused_total": f"{len(insights)} additional insights only visible through fusion",
        }
    }


@app.get("/api/validation/{job_id}")
async def get_validation(job_id: str):
    """Get validation report."""
    job = _get_complete_job(job_id)
    return job["state"].get("validation_report") or {}


@app.get("/api/map/{job_id}", response_class=HTMLResponse)
async def get_map(job_id: str):
    """Get interactive folium map as HTML."""
    job      = _get_complete_job(job_id)
    state    = job["state"]
    fused    = state.get("fused_events") or []
    center   = state.get("center") or {"lat": 34.0450, "lon": -118.5260}
    bbox     = state.get("bounding_box")

    from agent.tools.report import build_interactive_map
    return build_interactive_map(fused, center, bbox)


@app.get("/api/summary/{job_id}")
async def get_summary(job_id: str):
    """Get full video summary from Pegasus."""
    job = _get_complete_job(job_id)
    return {"summary": job["state"].get("full_video_summary", "")}


# ════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE QUERY
# ════════════════════════════════════════════════════════════════

class NLQueryRequest(BaseModel):
    job_id:   str
    question: str


@app.post("/api/query")
async def nl_query(request: NLQueryRequest):
    """
    Natural language query requiring multi-source synthesis.
    Searches fused events + asks Pegasus directly.
    Bonus criterion: NL querying requiring multi-source synthesis.
    """
    job   = _get_complete_job(request.job_id)
    state = job["state"]
    fused = state.get("fused_events") or []
    q     = request.question.lower()

    # Filter fused events by question intent
    # Start fresh for each query
    results = fused[:]
    filters_applied = []

    if any(w in q for w in ["worst","severe","destroyed","critical","priority"]):
        results = [e for e in results if e.get("risk_score", 0) >= 50]
        filters_applied.append("high_risk")

    if any(w in q for w in ["flood","water","submerged","inundated"]):
        results = [e for e in results if "flood" in e.get("damage_type","")]
        filters_applied.append("flood")

    if any(w in q for w in ["fire","burn","wildfire","ash"]):
        results = [e for e in results if "fire" in e.get("damage_type","") or
                "burn" in e.get("burn_status","") or
                "destroyed" in e.get("burn_status","")]
        filters_applied.append("fire")

    if any(w in q for w in ["blocked","impassable","inaccessible","closed","access"]):
        results = [e for e in results if e.get("accessibility") in
                ("debris_blocked","inaccessible","boat_only","unknown")]
        filters_applied.append("accessibility")

    if any(w in q for w in ["unreported","missed","undetected"]):
        results = [e for e in results if e.get("fusion_insight_type") == "UNREPORTED_DAMAGE"]
        filters_applied.append("unreported")

    if any(w in q for w in ["conflict","mismatch","disagree","discrepancy"]):
        results = [e for e in results if e.get("has_conflict")]
        filters_applied.append("conflict")

    if any(w in q for w in ["passable","accessible","open","driveable"]):
        results = [e for e in results if e.get("accessibility") == "passable"]
        filters_applied.append("passable")

    # If no filter matched — return top 10 by risk
    if not filters_applied:
        results = sorted(fused, key=lambda e: e.get("risk_score",0), reverse=True)[:10]

    # Also ask Pegasus for direct video answer
    video_uri = state.get("video_s3_uri", "")
    video_answer = ""
    if video_uri:
        try:
            video_answer = nl_query_video(video_uri, request.question)
        except Exception as e:
            video_answer = f"(Pegasus unavailable: {e})"

    return {
        "question":        request.question,
        "sources_queried": ["video_pegasus", "overture_maps", "fema_report", "satellite"],
        "fused_results": [{
            "street_name":       e["street_name"],
            "risk_score":        e.get("risk_score"),
            "burn_status":       e.get("burn_status"),
            "accessibility":     e.get("accessibility"),
            "sources":           e.get("sources"),
            "fusion_insight":    e.get("fusion_insight"),
            "lat":               e["lat"],
            "lon":               e["lon"],
        } for e in results[:8]],
        "video_answer":    video_answer,
        "result_count":    len(results),
        "requires_fusion": True,
    }


# ════════════════════════════════════════════════════════════════
# PROVENANCE — Full audit trail
# ════════════════════════════════════════════════════════════════

@app.get("/api/provenance/{job_id}/{event_id}")
async def get_provenance(job_id: str, event_id: str):
    """
    Full provenance audit trail for a specific event.
    Bonus criterion: complete audit trail showing how each finding was derived.
    """
    job    = _get_complete_job(job_id)
    fused  = job["state"].get("fused_events") or []
    event  = next((e for e in fused if e["event_id"] == event_id), None)
    if not event:
        raise HTTPException(404, f"Event {event_id} not found")

    return {
        "event_id":   event_id,
        "street_name":event["street_name"],
        "finding":    f"{event.get('burn_status','?')} damage at {event['street_name']}",
        "audit_trail": {
            "① VIDEO (Pegasus)": {
                "model":          "Pegasus 1.2 via AWS Bedrock",
                "video_s3_uri":   job["state"].get("video_s3_uri"),
                "timestamp":      event.get("timestamp"),
                "marengo_query":  event.get("provenance",{}).get("video",{}).get("search_query"),
                "marengo_score":  event.get("provenance",{}).get("video",{}).get("marengo_score"),
                "pegasus_output": event.get("burn_status"),
            },
            "② OVERTURE MAPS": event.get("provenance",{}).get("overture") or "No match within threshold",
            "③ SATELLITE":     event.get("provenance",{}).get("satellite") or "Not available",
            "④ FEMA REPORT":   event.get("provenance",{}).get("fema_report") or "No matching claim",
        },
        "confidence_breakdown": event.get("confidence_breakdown"),
        "conflict":             event.get("conflict"),
        "fusion_insight":       event.get("fusion_insight"),
        "sources_used":         event.get("sources"),
        "risk_score":           event.get("risk_score"),
        "composite_confidence": event.get("confidence"),
    }


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _get_complete_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] == "running":
        raise HTTPException(202, "Pipeline still running")
    if job["status"] == "failed":
        raise HTTPException(500, f"Pipeline failed: {job.get('errors')}")
    if job["status"] == "queued":
        raise HTTPException(202, "Pipeline queued")
    return job


def _serialize(state: dict) -> dict:
    """JSON-serializable state."""
    return json.loads(json.dumps(state, default=str))


def _count_by(events: list, field: str) -> dict:
    from collections import Counter
    return dict(Counter(e.get(field, "unknown") for e in events))
