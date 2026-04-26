"""
LangGraph Agent State
Full audit trail carried through every node.
Every tool output is appended — nothing is lost.
"""
from typing import Optional, Any
from typing_extensions import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    video_filename:      str
    video_s3_uri:        str
    video_local_path:    Optional[str]
    satellite_s3_uri:    Optional[str]
    nl_query:            Optional[str]        # user's natural language question
    scenario:            Optional[dict]       # matched VIDEO_SCENARIOS entry

    # ── Marengo ──────────────────────────────────────────────
    video_id:            Optional[str]
    embedding_data:      Optional[list]       # raw segment embeddings
    total_segments:      Optional[int]
    search_results:      Optional[list]       # Marengo NL search hits

    # ── Pegasus ──────────────────────────────────────────────
    streets:             Optional[list]       # parsed street assessments
    raw_pegasus_output:  Optional[str]        # raw text for parsing
    full_video_summary:  Optional[str]        # streaming summary

    # ── Geocoding ─────────────────────────────────────────────
    geocoded_streets:    Optional[list]       # streets with lat/lon
    bounding_box:        Optional[dict]
    center:              Optional[dict]

    # ── Overture Maps ─────────────────────────────────────────
    overture_buildings:  Optional[list]
    overture_roads:      Optional[list]

    # ── Satellite ─────────────────────────────────────────────
    satellite_baseline:  Optional[dict]

    # ── FEMA Reports ─────────────────────────────────────────
    fema_reports:        Optional[list]
    all_claims:          Optional[list]

    # ── Fusion Output ─────────────────────────────────────────
    fused_events:        Optional[list]
    fusion_insights:     Optional[list]
    conflicts_detected:  Optional[int]

    # ── Intelligence Products ─────────────────────────────────
    geojson:             Optional[dict]
    damage_table:        Optional[list]
    validation_report:   Optional[dict]
    mission_impact:      Optional[dict]
    map_html_path:       Optional[str]
    report_s3_uri:       Optional[str]

    # ── NL Query Response ────────────────────────────────────
    nl_answer:           Optional[str]

    # ── Pipeline Metadata ────────────────────────────────────
    pipeline_start:      Optional[str]
    processing_time_s:   Optional[float]
    errors:              Annotated[list, operator.add]  # accumulates errors
    completed_steps:     Annotated[list, operator.add]  # audit trail
    messages:            Annotated[list, operator.add]  # agent messages
