"""
Tool: Overture Maps — Real DuckDB Parquet Query
Pulls actual building footprints + road segments for any bbox.
Falls back to synthetic if DuckDB unavailable.
"""
import logging
import math
import random
import uuid

logger = logging.getLogger(__name__)

OVERTURE_BUCKET  = "overturemaps-us-west-2"
OVERTURE_RELEASE = "release/2024-09-18.0"


def query_overture_buildings(bbox: dict, max_results: int = 300) -> list:
    """
    Query real Overture Maps building footprints via DuckDB.
    Falls back to synthetic if DuckDB unavailable.
    """
    try:
        import duckdb
        return _duckdb_query_buildings(bbox, max_results)
    except ImportError:
        logger.warning("DuckDB not available — using synthetic Overture features")
        return _synthetic_buildings(bbox, max_results)
    except Exception as e:
        logger.warning(f"DuckDB query failed: {e} — using synthetic")
        return _synthetic_buildings(bbox, max_results)


def query_overture_roads(bbox: dict, max_results: int = 100) -> list:
    """
    Query real Overture Maps road segments via DuckDB.
    Falls back to synthetic if DuckDB unavailable.
    """
    try:
        import duckdb
        return _duckdb_query_roads(bbox, max_results)
    except ImportError:
        logger.warning("DuckDB not available — using synthetic road features")
        return _synthetic_roads(bbox, max_results)
    except Exception as e:
        logger.warning(f"DuckDB road query failed: {e} — using synthetic")
        return _synthetic_roads(bbox, max_results)


# ── Real DuckDB queries ───────────────────────────────────────

def _duckdb_query_buildings(bbox: dict, max_results: int) -> list:
    """Query Overture buildings Parquet via DuckDB httpfs."""
    import duckdb

    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_region='us-west-2';")

    min_lon = bbox["min_lon"]
    min_lat = bbox["min_lat"]
    max_lon = bbox["max_lon"]
    max_lat = bbox["max_lat"]

    prefix = f"s3://{OVERTURE_BUCKET}/{OVERTURE_RELEASE}/theme=buildings/type=building"

    query = f"""
        SELECT
            id,
            ST_X(ST_Centroid(geometry)) AS lon,
            ST_Y(ST_Centroid(geometry)) AS lat,
            class,
            subtype,
            height,
            num_floors,
            bbox.minx AS minx,
            bbox.miny AS miny,
            bbox.maxx AS maxx,
            bbox.maxy AS maxy
        FROM read_parquet('{prefix}/*', hive_partitioning=true)
        WHERE
            bbox.minx >= {min_lon} AND bbox.maxx <= {max_lon} AND
            bbox.miny >= {min_lat} AND bbox.maxy <= {max_lat}
        LIMIT {max_results}
    """

    rows    = conn.execute(query).fetchall()
    columns = ["id","lon","lat","class","subtype","height",
                "num_floors","minx","miny","maxx","maxy"]

    buildings = []
    for row in rows:
        r = dict(zip(columns, row))
        buildings.append({
            "overture_id":    str(r["id"]),
            "feature_type":   "building",
            "feature_class":  str(r.get("class") or r.get("subtype") or "residential"),
            "lat":            float(r["lat"]) if r["lat"] else None,
            "lon":            float(r["lon"]) if r["lon"] else None,
            "height_m":       float(r["height"]) if r.get("height") else None,
            "num_floors":     int(r["num_floors"]) if r.get("num_floors") else None,
            "pre_fire_status":"existed",
            "source":         "overture_real",
        })

    buildings = [b for b in buildings if b["lat"] and b["lon"]]
    logger.info(f"✅ Overture buildings (real): {len(buildings)}")
    return buildings


def _duckdb_query_roads(bbox: dict, max_results: int) -> list:
    """Query Overture road segments via DuckDB httpfs."""
    import duckdb

    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_region='us-west-2';")

    min_lon = bbox["min_lon"]
    min_lat = bbox["min_lat"]
    max_lon = bbox["max_lon"]
    max_lat = bbox["max_lat"]

    prefix = f"s3://{OVERTURE_BUCKET}/{OVERTURE_RELEASE}/theme=transportation/type=segment"

    query = f"""
        SELECT
            id,
            class,
            subtype,
            ST_X(ST_Centroid(geometry)) AS lon,
            ST_Y(ST_Centroid(geometry)) AS lat,
            bbox.minx AS minx,
            bbox.miny AS miny,
            bbox.maxx AS maxx,
            bbox.maxy AS maxy
        FROM read_parquet('{prefix}/*', hive_partitioning=true)
        WHERE
            bbox.minx >= {min_lon} AND bbox.maxx <= {max_lon} AND
            bbox.miny >= {min_lat} AND bbox.maxy <= {max_lat}
        LIMIT {max_results}
    """

    rows    = conn.execute(query).fetchall()
    columns = ["id","class","subtype","lon","lat","minx","miny","maxx","maxy"]

    roads = []
    for row in rows:
        r = dict(zip(columns, row))
        if r.get("lat") and r.get("lon"):
            roads.append({
                "overture_id":   str(r["id"]),
                "feature_type":  "road",
                "feature_class": str(r.get("class") or "road"),
                "lat":           float(r["lat"]),
                "lon":           float(r["lon"]),
                "pre_fire_status": "existed",
                "source":        "overture_real",
            })

    logger.info(f"✅ Overture roads (real): {len(roads)}")
    return roads


# ── Synthetic fallback ────────────────────────────────────────

def _synthetic_buildings(bbox: dict, n: int = 200) -> list:
    """Generate synthetic building features within bbox."""
    logger.warning(f"Using {n} synthetic Overture buildings")
    feats = []
    for i in range(n):
        lat = bbox["min_lat"] + random.random() * (bbox["max_lat"] - bbox["min_lat"])
        lon = bbox["min_lon"] + random.random() * (bbox["max_lon"] - bbox["min_lon"])
        feats.append({
            "overture_id":    f"synthetic_bldg_{i:04d}_{uuid.uuid4().hex[:6]}",
            "feature_type":   "building",
            "feature_class":  random.choice(["residential","commercial","retail"]),
            "lat":            lat,
            "lon":            lon,
            "height_m":       random.randint(4, 15),
            "num_floors":     random.randint(1, 3),
            "pre_fire_status":"existed",
            "source":         "synthetic",
        })
    return feats


def _synthetic_roads(bbox: dict, n: int = 50) -> list:
    """Generate synthetic road segments within bbox."""
    feats = []
    for i in range(n):
        lat = bbox["min_lat"] + random.random() * (bbox["max_lat"] - bbox["min_lat"])
        lon = bbox["min_lon"] + random.random() * (bbox["max_lon"] - bbox["min_lon"])
        feats.append({
            "overture_id":   f"synthetic_road_{i:04d}_{uuid.uuid4().hex[:6]}",
            "feature_type":  "road",
            "feature_class": random.choice(["primary","secondary","residential","service"]),
            "lat":           lat,
            "lon":           lon,
            "pre_fire_status": "existed",
            "source":        "synthetic",
        })
    return feats


# ── Spatial helpers ───────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R  = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def find_nearest(lat, lon, features, max_m=300, top_k=3) -> list:
    """Find nearest Overture features to a GPS point."""
    candidates = []
    for f in features:
        if not f.get("lat") or not f.get("lon"):
            continue
        d = haversine_m(lat, lon, f["lat"], f["lon"])
        if d <= max_m:
            candidates.append((d, f))
    candidates.sort(key=lambda x: x[0])
    return [(d, f) for d, f in candidates[:top_k]]


def build_geo_buckets(features: list, bucket_deg: float = 0.01) -> dict:
    """Spatial grid index — cheap pre-filter before haversine."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for f in features:
        if f.get("lat") and f.get("lon"):
            key = (round(f["lat"]/bucket_deg), round(f["lon"]/bucket_deg))
            buckets[key].append(f)
    return dict(buckets)


def get_nearby_from_bucket(lat, lon, buckets, bucket_deg=0.01, radius=2) -> list:
    """Get features from nearby grid buckets only."""
    bk = (round(lat/bucket_deg), round(lon/bucket_deg))
    nearby = []
    for dl in range(-radius, radius+1):
        for dm in range(-radius, radius+1):
            nearby.extend(buckets.get((bk[0]+dl, bk[1]+dm), []))
    return nearby
