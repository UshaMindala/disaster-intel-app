"""
LangGraph Agentic Pipeline — Disaster Intelligence System
Orchestrates all tools in sequence with state management.
Bonus criterion: Agent architecture orchestrating multi-source reasoning.
"""
import logging
from datetime import datetime
from typing import Literal
import uuid

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.tools.video import (
    upload_video, create_video_embedding,
    ensure_vector_index, index_segments, run_all_damage_queries
)
from agent.tools.pegasus import (
    emergency_assessment, full_video_summary, extract_streets_structured, nl_query_video
)
from agent.tools.overture import query_overture_buildings, query_overture_roads
from agent.tools.geocode_satellite_fema import (
    geocode_all_streets, analyze_satellite_image, generate_fema_reports
)
from agent.tools.fusion import (
    run_fusion, build_geojson, build_validation_report
)
from agent.tools.report import (
    build_interactive_map, build_mission_impact, save_all_outputs
)
from config import VIDEO_SCENARIOS

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# NODE DEFINITIONS — each node = one pipeline step
# ════════════════════════════════════════════════════════════════

def node_upload_video(state: AgentState) -> AgentState:
    """Node 1: Upload video to S3, detect scenario."""
    logger.info("▶ Node: upload_video")
    try:
        result   = upload_video(state["video_local_path"])
        scenario = result.get("scenario") or VIDEO_SCENARIOS.get(
            state.get("video_filename", ""), None
        )
        return {
            **state,
            "video_s3_uri":  result["video_s3_uri"],
            "video_filename":result["video_filename"],
            "scenario":      scenario,
            "pipeline_start":datetime.utcnow().isoformat(),
            "completed_steps":["upload_video"],
            "messages": [f"✅ Video uploaded: {result['video_s3_uri']}"],
        }
    except Exception as e:
        logger.error(f"upload_video failed: {e}")
        return {**state, "errors": [str(e)]}


def node_marengo_embed(state: AgentState) -> AgentState:
    """Node 2: Create Marengo video embeddings."""
    logger.info("▶ Node: marengo_embed")
    try:
        ensure_vector_index()
        result = create_video_embedding(state["video_s3_uri"])
        data   = result["embedding_data"]
        return {
            **state,
            "video_id":        result["video_id"],
            "embedding_data":  data,
            "total_segments":  len(data),
            "completed_steps": ["marengo_embed"],
            "messages": [f"✅ Marengo: {len(data)} clip segments"],
        }
    except Exception as e:
        logger.error(f"marengo_embed failed: {e}")
        # Don't stop pipeline — continue with empty embeddings
        return {
            **state,
            "video_id":        str(uuid.uuid4()),
            "embedding_data":  [],
            "total_segments":  0,
            "completed_steps": ["marengo_embed"],
            "errors":          [f"Marengo: {e}"],
            "messages":        [f"⚠️ Marengo embedding failed — continuing pipeline"],
        }


def node_marengo_index(state: AgentState) -> AgentState:
    """Node 3: Index embeddings in S3 Vectors."""
    logger.info("▶ Node: marengo_index")
    try:
        total = index_segments(
            state["embedding_data"],
            state["video_id"],
            state["video_s3_uri"],
        )
        return {
            **state,
            "completed_steps":["marengo_index"],
            "messages": [f"✅ S3 Vectors: {total} segments indexed"],
        }
    except Exception as e:
        logger.error(f"marengo_index failed: {e}")
        return {**state, "errors": [str(e)]}


def node_marengo_search(state: AgentState) -> AgentState:
    """Node 4: Run damage NL search queries via Marengo."""
    logger.info("▶ Node: marengo_search")
    try:
        hits = run_all_damage_queries(state.get("scenario"))
        return {
            **state,
            "search_results": hits,
            "completed_steps":["marengo_search"],
            "messages": [f"✅ Marengo search: {len(hits)} unique segments found"],
        }
    except Exception as e:
        logger.error(f"marengo_search failed: {e}")
        return {**state, "errors": [str(e)], "search_results": []}


def node_pegasus_analyze(state: AgentState) -> AgentState:
    logger.info("▶ Node: pegasus_analyze")
    try:
        summary = full_video_summary(state["video_s3_uri"])
        streets, raw = extract_streets_structured(state["video_s3_uri"])

        # If too few streets extracted, try emergency assessment too
        if len(streets) < 10:
            logger.warning(f"Only {len(streets)} streets — trying emergency assessment")
            try:
                emerg = emergency_assessment(state["video_s3_uri"])
                # Extract street names from priority list
                extra_streets = []
                for p in emerg.get("top_5_priorities", []):
                    loc = p.get("location","")
                    if loc and not any(s["street_name"] in loc for s in streets):
                        extra_streets.append({
                            "street_name":          loc,
                            "burn_status":          "mostly_destroyed",
                            "damage_type":          "fire",
                            "structures_destroyed": 10,
                            "structures_intact":    0,
                            "accessibility":        "inaccessible",
                            "priority_score":       p.get("rank", 5),
                            "hazards":              [],
                            "notable_observations": p.get("reason",""),
                            "timestamp":            "unknown",
                        })
                streets.extend(extra_streets)
                logger.info(f"Added {len(extra_streets)} streets from emergency assessment")
            except Exception as e2:
                logger.warning(f"Emergency assessment also failed: {e2}")

        return {
            **state,
            "full_video_summary": summary,
            "streets":            streets,
            "raw_pegasus_output": raw,
            "completed_steps":    ["pegasus_analyze"],
            "messages": [
                f"✅ Pegasus summary: {len(summary)} chars",
                f"✅ Streets extracted: {len(streets)}",
            ],
        }
    except Exception as e:
        logger.error(f"pegasus_analyze failed: {e}")
        return {**state, "errors": [str(e)], "streets": [],
                "completed_steps": ["pegasus_analyze"],
                "messages": [f"⚠️ Pegasus failed: {e}"]}


def node_geocode(state: AgentState) -> AgentState:
    """Node 6: Geocode all streets to lat/lon."""
    logger.info("▶ Node: geocode")
    try:
        streets, bbox, center = geocode_all_streets(
            state.get("streets", []),
            state.get("scenario"),
        )
        return {
            **state,
            "geocoded_streets": streets,
            "bounding_box":     bbox,
            "center":           center,
            "completed_steps":  ["geocode"],
            "messages": [f"✅ Geocoded: {sum(1 for s in streets if s.get('geocoded'))}/{len(streets)} streets"],
        }
    except Exception as e:
        logger.error(f"geocode failed: {e}")
        return {**state, "errors": [str(e)]}


def node_overture(state: AgentState) -> AgentState:
    """Node 7: Query Overture Maps for pre-disaster features."""
    logger.info("▶ Node: overture")
    try:
        bbox      = state.get("bounding_box") or (
            state["scenario"]["bbox"] if state.get("scenario") else {}
        )
        buildings = query_overture_buildings(bbox, max_results=300)
        roads     = query_overture_roads(bbox, max_results=100)
        return {
            **state,
            "overture_buildings": buildings,
            "overture_roads":     roads,
            "completed_steps":    ["overture"],
            "messages": [
                f"✅ Overture: {len(buildings)} buildings, {len(roads)} roads",
                f"   Source: {'real DuckDB' if buildings and buildings[0].get('source')=='overture_real' else 'synthetic'}",
            ],
        }
    except Exception as e:
        logger.error(f"overture failed: {e}")
        return {**state, "errors": [str(e)],
                "overture_buildings": [], "overture_roads": []}


def node_satellite(state: AgentState) -> AgentState:
    """Node 8: Analyze pre-fire satellite image."""
    logger.info("▶ Node: satellite")
    try:
        if not state.get("satellite_s3_uri"):
            logger.info("  No satellite image — skipping")
            return {
                **state,
                "satellite_baseline": None,
                "completed_steps":    ["satellite_skipped"],
                "messages": ["ℹ️  Satellite: no image provided — skipping"],
            }

        baseline = analyze_satellite_image(state["satellite_s3_uri"])
        return {
            **state,
            "satellite_baseline": baseline,
            "completed_steps":    ["satellite"],
            "messages": [
                f"✅ Satellite baseline: {baseline.get('total_structures','?')} structures",
                f"   Vegetation: {baseline.get('vegetation_coverage_pct','?')}%",
            ],
        }
    except Exception as e:
        logger.error(f"satellite failed: {e}")
        return {**state, "errors": [str(e)], "satellite_baseline": None}


def node_fema_reports(state: AgentState) -> AgentState:
    """Node 9: Generate synthetic FEMA damage reports."""
    logger.info("▶ Node: fema_reports")
    try:
        reports, claims = generate_fema_reports(
            scenario=state.get("scenario"), count=4
        )
        return {
            **state,
            "fema_reports":    reports,
            "all_claims":      claims,
            "completed_steps": ["fema_reports"],
            "messages": [f"✅ FEMA: {len(reports)} reports | {len(claims)} claims"],
        }
    except Exception as e:
        logger.error(f"fema_reports failed: {e}")
        return {**state, "errors": [str(e)],
                "fema_reports": [], "all_claims": []}


def node_fusion(state: AgentState) -> AgentState:
    """Node 10: Run multi-source fusion engine."""
    logger.info("▶ Node: fusion")
    try:
        streets = state.get("geocoded_streets") or state.get("streets", [])
        fused, insights, conflicts = run_fusion(
            streets            = streets,
            overture_buildings = state.get("overture_buildings", []),
            overture_roads     = state.get("overture_roads", []),
            satellite_baseline = state.get("satellite_baseline"),
            fema_reports       = state.get("fema_reports", []),
            all_claims         = state.get("all_claims", []),
            scenario           = state.get("scenario"),
        )
        return {
            **state,
            "fused_events":      fused,
            "fusion_insights":   insights,
            "conflicts_detected":conflicts,
            "completed_steps":   ["fusion"],
            "messages": [
                f"✅ Fusion: {len(fused)} events",
                f"   Insights: {len(insights)} | Conflicts: {conflicts}",
                f"   Multi-source: {sum(1 for e in fused if e.get('source_count',1)>=2)}",
            ],
        }
    except Exception as e:
        logger.error(f"fusion failed: {e}")
        return {**state, "errors": [str(e)], "fused_events": []}


def node_generate_products(state: AgentState) -> AgentState:
    """Node 11: Generate all intelligence products."""
    logger.info("▶ Node: generate_products")
    try:
        fused    = state.get("fused_events", [])
        center   = state.get("center") or (
            state["scenario"]["center"] if state.get("scenario") else {"lat":0,"lon":0}
        )
        scenario = state.get("scenario")

        # GeoJSON
        geojson  = build_geojson(fused, scenario)

        # Validation report
        start    = datetime.fromisoformat(state.get("pipeline_start", datetime.utcnow().isoformat()))
        proc_s   = (datetime.utcnow() - start).total_seconds()
        validation = build_validation_report(fused, state.get("all_claims",[]), proc_s)

        # Mission impact brief
        impact   = build_mission_impact(fused, proc_s, scenario)

        # Interactive map
        map_html = build_interactive_map(fused, center, state.get("bounding_box"))

        # Save all to S3
        uris     = save_all_outputs(fused, geojson, validation, impact, map_html, scenario)

        return {
            **state,
            "geojson":            geojson,
            "damage_table":       fused,
            "validation_report":  validation,
            "mission_impact":     impact,
            "map_html_path":      uris.get("map_html"),
            "report_s3_uri":      uris.get("full_assessment"),
            "processing_time_s":  proc_s,
            "completed_steps":    ["generate_products"],
            "messages": [
                f"✅ Products: GeoJSON + Map + Validation + Mission Impact",
                f"   Report: {uris.get('full_assessment')}",
                f"   Processing time: {proc_s:.1f}s",
            ],
        }
    except Exception as e:
        logger.error(f"generate_products failed: {e}")
        return {**state, "errors": [str(e)]}


def node_nl_query(state: AgentState) -> AgentState:
    """Node 12: Answer NL query using Pegasus + fused events."""
    logger.info("▶ Node: nl_query")
    try:
        question = state.get("nl_query", "")
        if not question:
            return {**state, "nl_answer": "", "completed_steps": ["nl_query_skipped"]}

        # First: try to answer from fused events
        fused    = state.get("fused_events", [])
        answer   = _answer_from_fused(question, fused)

        # If insufficient, ask Pegasus directly
        if len(answer) < 100:
            pegasus_answer = nl_query_video(state["video_s3_uri"], question)
            answer = f"{answer}\n\nVideo Analysis:\n{pegasus_answer}"

        return {
            **state,
            "nl_answer":       answer,
            "completed_steps": ["nl_query"],
            "messages": [f"✅ NL query answered: {len(answer)} chars"],
        }
    except Exception as e:
        logger.error(f"nl_query failed: {e}")
        return {**state, "errors": [str(e)], "nl_answer": "Query failed — see logs."}


def _answer_from_fused(question: str, fused_events: list) -> str:
    """Answer NL question from fused event data."""
    q = question.lower()
    results = fused_events[:]

    # Filter based on question intent
    if any(w in q for w in ["worst","highest","priority","critical","severe","destroyed"]):
        results = [e for e in results if e.get("risk_score", 0) >= 60]
    if any(w in q for w in ["flood","water","submerged"]):
        results = [e for e in results if "flood" in e.get("damage_type","")]
    if any(w in q for w in ["fire","burn","wildfire"]):
        results = [e for e in results if "fire" in e.get("damage_type","") or
                   "burn" in e.get("burn_status","")]
    if any(w in q for w in ["road","route","access","blocked"]):
        results = [e for e in results if e.get("accessibility") in
                   ("debris_blocked","inaccessible","boat_only")]
    if any(w in q for w in ["unreported","missed","missing"]):
        results = [e for e in results if e.get("fusion_insight_type") == "UNREPORTED_DAMAGE"]
    if any(w in q for w in ["conflict","disagree","mismatch"]):
        results = [e for e in results if e.get("has_conflict")]

    results.sort(key=lambda e: e.get("risk_score", 0), reverse=True)

    if not results:
        return ""

    lines = [f"Found {len(results)} matching location(s):\n"]
    for e in results[:5]:
        lines.append(
            f"• {e['street_name']} — {e.get('burn_status','?').replace('_',' ').upper()} "
            f"| Risk: {e.get('risk_score',0)}/100 | Confidence: {e.get('confidence',0):.1f}/100\n"
            f"  Sources: {', '.join(e.get('sources',[]))}\n"
            + (f"  🔍 {e['fusion_insight']}\n" if e.get('fusion_insight') else "")
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# ROUTING LOGIC
# ════════════════════════════════════════════════════════════════

def route_after_upload(state: AgentState) -> Literal["marengo_embed", "end"]:
    if state.get("errors"):
        return "end"
    return "marengo_embed"


def route_after_products(state: AgentState) -> Literal["nl_query", "end"]:
    if state.get("nl_query"):
        return "nl_query"
    return "end"


# ════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("upload_video",        node_upload_video)
    graph.add_node("marengo_embed",       node_marengo_embed)
    graph.add_node("marengo_index",       node_marengo_index)
    graph.add_node("marengo_search",      node_marengo_search)
    graph.add_node("pegasus_analyze",     node_pegasus_analyze)
    graph.add_node("geocode",             node_geocode)
    graph.add_node("overture",            node_overture)
    graph.add_node("satellite",           node_satellite)
    graph.add_node("fema_reports",        node_fema_reports)
    graph.add_node("fusion",              node_fusion)
    graph.add_node("generate_products",   node_generate_products)
    graph.add_node("nl_query",            node_nl_query)

    # Entry point
    graph.set_entry_point("upload_video")

    # Edges — sequential pipeline
    graph.add_conditional_edges("upload_video", route_after_upload,
                                {"marengo_embed":"marengo_embed","end":END})
    graph.add_edge("marengo_embed",     "marengo_index")
    graph.add_edge("marengo_index",     "marengo_search")
    graph.add_edge("marengo_search",    "pegasus_analyze")
    graph.add_edge("pegasus_analyze",   "geocode")
    graph.add_edge("geocode",           "overture")
    graph.add_edge("overture",          "satellite")
    graph.add_edge("satellite",         "fema_reports")
    graph.add_edge("fema_reports",      "fusion")
    graph.add_edge("fusion",            "generate_products")
    graph.add_conditional_edges("generate_products", route_after_products,
                                {"nl_query":"nl_query","end":END})
    graph.add_edge("nl_query", END)

    return graph.compile()


# Singleton compiled graph
pipeline = build_graph()


def run_pipeline(video_local_path: str,
                 satellite_s3_uri: str = None,
                 nl_query: str = None) -> AgentState:
    """Run the full agentic pipeline."""
    import os
    initial_state: AgentState = {
        "video_filename":     os.path.basename(video_local_path),
        "video_s3_uri":       "",
        "video_local_path":   video_local_path,
        "satellite_s3_uri":   satellite_s3_uri,
        "nl_query":           nl_query,
        "scenario":           None,
        "video_id":           None,
        "embedding_data":     None,
        "total_segments":     None,
        "search_results":     None,
        "streets":            None,
        "raw_pegasus_output": None,
        "full_video_summary": None,
        "geocoded_streets":   None,
        "bounding_box":       None,
        "center":             None,
        "overture_buildings": None,
        "overture_roads":     None,
        "satellite_baseline": None,
        "fema_reports":       None,
        "all_claims":         None,
        "fused_events":       None,
        "fusion_insights":    None,
        "conflicts_detected": None,
        "geojson":            None,
        "damage_table":       None,
        "validation_report":  None,
        "mission_impact":     None,
        "map_html_path":      None,
        "report_s3_uri":      None,
        "nl_answer":          None,
        "pipeline_start":     datetime.utcnow().isoformat(),
        "processing_time_s":  None,
        "errors":             [],
        "completed_steps":    [],
        "messages":           [],
    }
    return pipeline.invoke(initial_state)
