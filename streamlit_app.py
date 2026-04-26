"""
Disaster Intelligence System — Streamlit App
Multi-source AI damage assessment powered by AWS Bedrock + LangGraph
"""
import os
import sys
import tempfile
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Intelligence System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
    .stAlert { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("Disaster Intel")
    st.caption("AWS Bedrock · LangGraph · 4-Source Fusion")
    st.divider()

    video_file = st.file_uploader(
        "Disaster Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload hurricane, wildfire, or flood footage",
    )
    sat_file = st.file_uploader(
        "Satellite Image (optional)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Pre-disaster satellite image for baseline analysis",
    )
    nl_query_input = st.text_input(
        "Analysis Question (optional)",
        placeholder="Which streets are inaccessible?",
    )

    st.divider()
    run_btn = st.button(
        "Run Analysis",
        disabled=(video_file is None),
        use_container_width=True,
        type="primary",
    )
    if st.session_state.result:
        if st.button("Clear Results", use_container_width=True):
            st.session_state.result = None
            st.rerun()


# ── Satellite upload helper ────────────────────────────────────
def upload_satellite(sat_file_obj) -> Optional[str]:
    """Upload satellite image to S3, return s3:// URI or None on failure."""
    try:
        import boto3
        from config import S3_BUCKET, S3_SATELLITE_PFX, AWS_REGION

        suffix = Path(sat_file_obj.name).suffix or ".jpg"
        key = f"{S3_SATELLITE_PFX}/sat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{suffix}"
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=sat_file_obj.read(),
        )
        return f"s3://{S3_BUCKET}/{key}"
    except Exception as e:
        st.warning(f"Satellite upload failed — running without satellite baseline. ({e})")
        return None


# ── Pipeline runner (streaming) ────────────────────────────────
def run_analysis(video_path: str, sat_s3_uri: Optional[str], query: Optional[str]):
    """Stream the LangGraph pipeline with live status updates."""
    from agent.graph import pipeline as langgraph_pipeline

    initial_state = {
        "video_filename":     os.path.basename(video_path),
        "video_s3_uri":       "",
        "video_local_path":   video_path,
        "satellite_s3_uri":   sat_s3_uri,
        "nl_query":           query or None,
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

    step_labels = {
        "upload_video":      "Upload Video to S3",
        "marengo_embed":     "Marengo Video Embeddings",
        "marengo_index":     "Index in S3 Vectors",
        "marengo_search":    "Damage Query Search",
        "pegasus_analyze":   "Pegasus Street Analysis",
        "geocode":           "Geocoding Streets",
        "overture":          "Overture Maps Baseline",
        "satellite":         "Satellite Analysis",
        "satellite_skipped": "Satellite (Skipped)",
        "fema_reports":      "FEMA Reports",
        "fusion":            "Multi-Source Fusion",
        "generate_products": "Generate Intelligence Products",
        "nl_query":          "Natural Language Query",
        "nl_query_skipped":  "NL Query (Skipped)",
    }

    final_state = None
    shown_msgs: set = set()

    with st.status("Starting intelligence pipeline...", expanded=True) as status:
        try:
            for state_snapshot in langgraph_pipeline.stream(initial_state, stream_mode="values"):
                final_state = state_snapshot

                steps    = state_snapshot.get("completed_steps") or []
                messages = state_snapshot.get("messages") or []
                errors   = state_snapshot.get("errors") or []

                if steps:
                    last  = steps[-1]
                    label = step_labels.get(last, last.replace("_", " ").title())
                    status.update(label=f"Step {len(steps)}/12 — {label}")

                for msg in messages:
                    if msg not in shown_msgs:
                        shown_msgs.add(msg)
                        if msg.startswith("⚠️") or msg.startswith("❌"):
                            st.warning(msg)
                        else:
                            st.write(msg)

                for err in errors:
                    key = f"ERR:{err}"
                    if key not in shown_msgs:
                        shown_msgs.add(key)
                        st.error(f"⚠️ {err[:300]}")

            status.update(label="Analysis complete!", state="complete", expanded=False)

        except Exception as exc:
            import traceback
            status.update(label="Pipeline failed", state="error")
            st.error(str(exc))
            st.code(traceback.format_exc())

    return final_state


# ── Trigger analysis on button click ──────────────────────────
if run_btn and video_file is not None:
    st.session_state.result = None

    # Save video to a temp file (keep original stem for scenario detection)
    suffix = Path(video_file.name).suffix or ".mp4"
    stem   = Path(video_file.name).stem
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=stem + "_") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    # Upload satellite if provided
    sat_s3_uri = upload_satellite(sat_file) if sat_file else None

    try:
        result = run_analysis(video_path, sat_s3_uri, nl_query_input or None)
        if result:
            st.session_state.result = dict(result)
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass

    st.rerun()


# ── Main content area ─────────────────────────────────────────
result = st.session_state.result

# ── Welcome screen ─────────────────────────────────────────────
if result is None:
    st.title("Disaster Intelligence System")
    st.markdown(
        "Upload a disaster video to run the **12-node agentic intelligence pipeline** — "
        "fusing 4 independent data sources into actionable multi-source damage assessments."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Video Analysis**\nBedrock Marengo embeddings + Pegasus street-level damage extraction")
    c2.info("**Geospatial Baseline**\nOverture Maps building & road footprints via DuckDB Parquet")
    c3.info("**FEMA Reports**\nTemporal claim matching with safety-conservative conflict resolution")
    c4.info("**Fusion Engine**\n4-axis confidence scoring · 5 insight types · full provenance trail")

    st.divider()
    st.subheader("Known Scenarios")
    s1, s2, s3 = st.columns(3)
    s1.success("Hurricane Ian · Fort Myers Beach, FL · 2022-09-28\n`zJyDF8_NHcs.mp4`")
    s2.warning("Palisades Fire · Pacific Palisades, LA · 2025-01-07\n`Palisades_Wildfire.mp4`")
    s3.info("Generic fallback · Any disaster video\n(11 general damage queries)")

# ── Dashboard ──────────────────────────────────────────────────
else:
    events   = result.get("fused_events") or []
    insights = result.get("fusion_insights") or []
    val      = result.get("validation_report") or {}
    impact   = result.get("mission_impact") or {}
    errors   = result.get("errors") or []
    proc_s   = result.get("processing_time_s") or 0
    scenario = result.get("scenario") or {}
    nl_ans   = result.get("nl_answer") or ""

    # Derived stats
    destroyed = sum(
        1 for e in events
        if (e.get("burn_status") or "").replace("_", "") in
           ("destroyed", "completelydestroyed", "mostlydestroyed")
    )
    multi_src = sum(
        1 for e in events
        if isinstance(e.get("sources"), list) and len(e["sources"]) >= 2
    )
    conflicts = result.get("conflicts_detected") or 0
    avg_risk  = sum(e.get("risk_score", 0) for e in events) / max(len(events), 1)
    scen_name = (
        scenario.get("name") or scenario.get("location") or "Unknown"
        if isinstance(scenario, dict) else "Unknown"
    )

    st.title(f"Disaster Intelligence — {scen_name}")
    st.caption(f"Processed in {proc_s:.1f}s · {len(events)} fused events · {len(insights)} insights")

    # ── Stat cards ─────────────────────────────────────────────
    mc = st.columns(6)
    mc[0].metric("Total Events", len(events))
    mc[1].metric("Destroyed", destroyed)
    mc[2].metric("Fusion Insights", len(insights))
    mc[3].metric("Conflicts", conflicts)
    mc[4].metric("Multi-Source", multi_src)
    mc[5].metric("Avg Risk", f"{avg_risk:.0f}/100")

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────
    t_ev, t_ins, t_map, t_q, t_val = st.tabs(
        ["Events", "Insights", "Map", "Query", "Validation & Impact"]
    )

    # ── Events tab ─────────────────────────────────────────────
    with t_ev:
        fc1, _, fc3 = st.columns([3, 2, 1])
        with fc3:
            view = st.selectbox(
                "Filter",
                ["All", "High Risk (≥60)", "Fusion Insights", "Destroyed", "Conflicts"],
                label_visibility="collapsed",
            )

        shown = events
        if view == "High Risk (≥60)":
            shown = [e for e in events if (e.get("risk_score") or 0) >= 60]
        elif view == "Fusion Insights":
            shown = [e for e in events if e.get("fusion_insight_type") or e.get("fusion_insight")]
        elif view == "Destroyed":
            shown = [
                e for e in events
                if (e.get("burn_status") or "").replace("_","") in
                   ("destroyed","completelydestroyed","mostlydestroyed")
            ]
        elif view == "Conflicts":
            shown = [e for e in events if e.get("has_conflict")]

        with fc1:
            st.caption(f"Showing {len(shown)} of {len(events)} events")

        if shown:
            import pandas as pd

            rows = []
            for e in shown:
                srcs = e.get("sources") or []
                rows.append({
                    "Location":      e.get("street_name", "Unknown"),
                    "Status":        (e.get("burn_status") or "unknown").replace("_", " ").title(),
                    "Risk":          int(e.get("risk_score") or 0),
                    "Confidence":    round(float(e.get("confidence") or e.get("composite_confidence") or 0), 1),
                    "Accessibility": (e.get("accessibility") or e.get("road_accessible") or "?").replace("_", " "),
                    "Destroyed":     int(e.get("structures_destroyed") or 0),
                    "Insight":       (e.get("fusion_insight_type") or ""),
                    "Sources":       len(srcs) if isinstance(srcs, list) else (e.get("source_count") or 1),
                    "Lat":           round(float(e.get("lat") or 0), 5),
                    "Lon":           round(float(e.get("lon") or 0), 5),
                })

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Risk": st.column_config.ProgressColumn(
                        "Risk", min_value=0, max_value=100, format="%d"
                    ),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.1f"),
                    "Sources":    st.column_config.NumberColumn("Sources", format="%d"),
                },
            )

            # Provenance expander for first high-risk event
            top = max(shown, key=lambda e: e.get("risk_score") or 0, default=None)
            if top and top.get("provenance"):
                with st.expander(f"Provenance audit — {top.get('street_name','?')} (highest risk)"):
                    prov = top["provenance"]
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown("**① Video (Pegasus/Marengo)**")
                        video_p = prov.get("video") or {}
                        st.json(video_p, expanded=False)
                        st.markdown("**② Overture Maps**")
                        st.json(prov.get("overture") or {}, expanded=False)
                    with pc2:
                        st.markdown("**③ Satellite**")
                        st.json(prov.get("satellite") or {}, expanded=False)
                        st.markdown("**④ FEMA Report**")
                        st.json(prov.get("fema_report") or {}, expanded=False)
        else:
            st.info("No events match the selected filter.")

    # ── Insights tab ───────────────────────────────────────────
    with t_ins:
        if insights:
            INSIGHT_ICONS = {
                "CONFIRMED_TOTAL_LOSS":        "🔴",
                "ACCESSIBLE_DESTRUCTION_ZONE": "🟠",
                "HIGH_FUEL_LOAD_DESTRUCTION":  "🟡",
                "SEVERITY_DISCREPANCY":        "🟣",
                "UNREPORTED_DAMAGE":           "⚫",
            }

            for ins in insights:
                itype  = ins.get("type") or ins.get("insight_type") or "INSIGHT"
                street = ins.get("street") or ins.get("location") or "Unknown"
                desc   = ins.get("description") or ""
                action = ins.get("actionability") or ins.get("action") or ""
                srcs   = ins.get("sources") or []

                icon  = INSIGHT_ICONS.get(itype, "🔵")
                label = itype.replace("_", " ")

                with st.expander(f"{icon} **{label}** — {street}", expanded=True):
                    if desc:
                        st.write(desc)
                    if action:
                        st.info(f"**Action:** {action}")
                    if srcs:
                        st.caption(f"Sources: {', '.join(srcs)}")
        else:
            st.info("No fusion insights detected for this video.")

    # ── Map tab ────────────────────────────────────────────────
    with t_map:
        map_html = None

        # Try S3 first
        s3_uri = result.get("map_html_path") or ""
        if s3_uri.startswith("s3://"):
            try:
                import boto3
                from config import AWS_REGION
                m = re.match(r"s3://([^/]+)/(.+)", s3_uri)
                if m:
                    bkt, key = m.groups()
                    s3c = boto3.client("s3", region_name=AWS_REGION)
                    obj = s3c.get_object(Bucket=bkt, Key=key)
                    map_html = obj["Body"].read().decode("utf-8")
            except Exception as e:
                st.caption(f"S3 map unavailable ({e}) — rebuilding locally.")

        # Rebuild locally from fused events
        if not map_html and events:
            try:
                from agent.tools.report import build_interactive_map

                center = result.get("center")
                if not center:
                    lats = [e.get("lat") for e in events if e.get("lat")]
                    lons = [e.get("lon") for e in events if e.get("lon")]
                    if lats:
                        center = {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}
                    else:
                        center = {"lat": 0.0, "lon": 0.0}

                map_html = build_interactive_map(events, center, result.get("bounding_box"))
            except Exception as e:
                st.error(f"Could not build map: {e}")

        if map_html:
            st.components.v1.html(map_html, height=600)
        else:
            st.info("No map available. Run the pipeline to generate a map.")

    # ── Query tab ──────────────────────────────────────────────
    with t_q:
        if nl_ans:
            st.subheader("Pipeline Query Result")
            st.write(nl_ans)
            st.divider()

        st.subheader("Ask About the Analysis")
        qc1, qc2 = st.columns([4, 1])
        with qc1:
            query_text = st.text_input(
                "Question",
                placeholder="Which areas have unreported damage?",
                label_visibility="collapsed",
                key="live_query",
            )
        with qc2:
            ask_btn = st.button("Search", use_container_width=True)

        if ask_btn and query_text:
            if events:
                from agent.graph import _answer_from_fused
                answer = _answer_from_fused(query_text, events)
                if answer:
                    st.markdown(answer)
                else:
                    st.info("No matching events found for that query.")
            else:
                st.warning("No fused events available to query.")

        st.caption(
            "Tip: Try keywords like *worst*, *flood*, *fire*, *blocked*, "
            "*unreported*, *conflict*, *access*"
        )

    # ── Validation & Impact tab ─────────────────────────────────
    with t_val:
        vc1, vc2 = st.columns(2)

        with vc1:
            st.subheader("Validation Metrics")
            if val:
                precision = float(val.get("precision") or val.get("event_precision") or 0)
                recall    = float(val.get("recall") or val.get("event_recall") or 0)
                f1        = float(val.get("f1") or val.get("f1_score") or 0)

                vm = st.columns(3)
                vm[0].metric("Precision", f"{precision:.1%}")
                vm[1].metric("Recall",    f"{recall:.1%}")
                vm[2].metric("F1 Score",  f"{f1:.1%}")

                st.caption(f"Processing time: {proc_s:.1f}s · Events: {len(events)}")

                skip = {"precision","recall","f1","event_precision","event_recall","f1_score","events","metrics"}
                for k, v in val.items():
                    if k not in skip and not isinstance(v, (list, dict)):
                        st.write(f"**{k.replace('_',' ').title()}:** {v}")
            else:
                st.info("No validation data available.")

        with vc2:
            st.subheader("Mission Impact")
            if impact:
                missions = impact.get("applicable_missions") or []
                skip     = {"applicable_missions"}
                for k, v in impact.items():
                    if k not in skip and not isinstance(v, (list, dict)):
                        st.write(f"**{k.replace('_',' ').title()}:** {v}")
                if missions:
                    st.write("**Applicable Missions:**")
                    for m in missions:
                        st.write(f"• {m}")
            else:
                st.info("No mission impact data available.")

        # GeoJSON download
        geojson = result.get("geojson")
        if geojson:
            import json
            st.divider()
            st.download_button(
                label="Download GeoJSON (FEMA format)",
                data=json.dumps(geojson, indent=2),
                file_name="damage_assessment.geojson",
                mime="application/geo+json",
            )

        # Pipeline errors
        if errors:
            with st.expander(f"Pipeline Errors ({len(errors)})"):
                for err in errors:
                    st.error(err)
