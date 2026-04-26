# 🛰️ Disaster Intelligence System
**Hackathon Track 03 · Multimodal Geospatial Workloads**  
Multi-Source Intelligence Fusion · TwelveLabs on AWS Bedrock · LangGraph Agent

---

## What It Does

Upload any disaster video (hurricane, wildfire, flood) → the agent automatically:
1. Embeds video with **Marengo 3.0** → indexes in **S3 Vectors**
2. Runs 10+ damage NL search queries → finds timestamped segments
3. Analyzes every segment with **Pegasus 1.2** → structured damage data
4. Geocodes all streets via Nominatim → lat/lon per finding
5. Queries **Overture Maps** (real DuckDB Parquet) → pre-disaster building counts
6. Analyzes pre-disaster satellite image (Google Earth Pro screenshot) → baseline
7. Generates synthetic **FEMA reports** → 3rd modality for conflict detection
8. Runs **fusion engine** → correlates all sources, detects conflicts, generates insights
9. Produces all **required deliverables** → GeoJSON, map, validation report, mission impact

---

## Architecture

```
Video Upload
     ↓
LangGraph Agent (12 nodes)
     ↓
┌─────────────────────────────────────────────┐
│  tool_video      → S3 + Marengo embed       │
│  tool_index      → S3 Vectors               │
│  tool_search     → Marengo NL queries       │
│  tool_pegasus    → Structured analysis      │
│  tool_geocode    → Nominatim                │
│  tool_overture   → DuckDB Parquet           │
│  tool_satellite  → Pegasus image analysis   │
│  tool_fema       → Synthetic reports        │
│  tool_fusion     → 4-axis confidence        │
│  tool_report     → All output products      │
└─────────────────────────────────────────────┘
     ↓
Intelligence Products:
  GeoJSON (FEMA) · Map · Validation · Mission Impact
```

---

## Quick Start

### 1. Install
```bash
conda activate geo-ai
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
copy .env.example .env
# Edit .env with your AWS credentials
```

Or set environment variables directly:
```powershell
$env:AWS_DEFAULT_REGION="us-east-1"
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
$env:AWS_SESSION_TOKEN="your_token"
```

### 3. Run
```bash
cd disaster-intel-agent
uvicorn app.main:app --reload --port 8000
```

### 4. Open browser
```
http://localhost:8000
```

### 5. Use the app
1. Upload disaster video (hurricane/wildfire/flood)
2. Optionally upload pre-disaster satellite screenshot
3. Click **Analyze Disaster Video**
4. Watch pipeline steps complete in real time
5. Explore damage events, fusion insights, map, NL query

---

## AWS Services Used

| Service | Purpose |
|---|---|
| Amazon Bedrock (Marengo 3.0) | Video embeddings |
| Amazon Bedrock (Pegasus 1.2) | Video-to-text damage analysis |
| S3 | Video, clips, outputs storage |
| S3 Vectors | Vector store for embeddings |
| STS | Credential resolution |

---

## Judging Criteria Coverage

| Criterion | Weight | Implementation |
|---|---|---|
| Multi-Source Integration | 30% | Video + Overture + Satellite + FEMA fused |
| Intelligence Value | 25% | 5 fusion insight types, single-source comparison |
| Video Understanding | 20% | Marengo embed + S3 Vectors + Pegasus analysis |
| System Design | 15% | LangGraph agent, staged pipeline, error handling |
| Technical Execution | 10% | FastAPI, clean modules, reproducible |

## Bonus Points
- ✅ 3+ modalities (video + Overture + satellite + FEMA = 4)
- ✅ Provenance tracking (full audit trail per event)
- ✅ Conflict resolution (video vs report severity)
- ✅ NL querying (multi-source synthesis)
- ✅ Agent architecture (LangGraph 12-node pipeline)

---

## Supported Videos
- `zJyDF8_NHcs.mp4` — Hurricane Ian, Fort Myers Beach / Cape Coral FL
- `Palisades_Wildfire.mp4` — Palisades Fire, Pacific Palisades LA
- Any disaster video — generic queries apply automatically
