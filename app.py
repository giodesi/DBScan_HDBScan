"""
DBSCAN & HDBSCAN clustering explorer
====================================

A general, interactive version of the notebook "Comparing DBSCAN and HDBSCAN
clustering" (ODCAF museums across Canada).

The notebook loads one CSV, keeps the museums, selects Latitude/Longitude,
removes the '..' placeholders, doubles the latitude, runs DBSCAN and HDBSCAN
and plots both results on a basemap of Canada. This app runs the same pipeline
on any CSV file:

* load a CSV/TSV (also inside a .zip or .gz) from an upload, a URL or the
  app's data/ folder, with automatic encoding, delimiter and decimal detection
  and configurable missing-value markers;
* explore the table (head, column summary, value counts);
* filter rows on any column, choose numeric (and optional one-hot encoded
  categorical) features, convert them to numbers and drop incomplete rows;
* scale them (none, z-score, min-max, robust, per-feature weights) or use the
  haversine great-circle distance for latitude/longitude data;
* run DBSCAN (with a k-distance plot that suggests eps) and HDBSCAN (the
  hdbscan library used in the notebook, or scikit-learn);
* inspect the clusters on an interactive map or in feature space, plus the
  notebook-style static map on a GeoTIFF (e.g. Canada.tif) or online basemap;
* compare both models and export the labelled data and a Python script that
  reproduces the run.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import codecs
import csv
import gzip
import hashlib
import inspect
import io
import math
import re
import tempfile
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:  # the HDBSCAN implementation used in the notebook
    import hdbscan as hdbscan_lib
except Exception:  # optional dependency
    hdbscan_lib = None

try:  # scikit-learn >= 1.3 ships its own HDBSCAN
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
except Exception:
    SkHDBSCAN = None

try:  # notebook-style static map
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:  # basemaps for the static map (as in the notebook)
    import contextily as ctx
except Exception:
    ctx = None


# =============================================================================
# Constants
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EARTH_RADIUS_KM = 6371.0088
WEB_MERCATOR_RADIUS = 6378137.0
# Link given in the notebook for the ODCAF archive (Statistics Canada).
ODCAF_ZIP_URL = "https://www150.statcan.gc.ca/n1/en/pub/21-26-0001/2020001/ODCAF_V1.0.zip?st=brOCT3Ry"

TABULAR_EXTENSIONS = (".csv", ".tsv", ".txt", ".tab", ".dat")
FOLDER_EXTENSIONS = (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".zip")
DEFAULT_MISSING_MARKERS = "..,?"
ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16"]
DELIMITERS = {",": "comma  ,", ";": "semicolon  ;", "\t": "tab", "|": "pipe  |", r"\s+": "whitespace"}

MAX_FILTER_OPTIONS = 1000
LARGE_DATA_ROWS = 20_000
DEFAULT_SAMPLE_ROWS = 20_000
MAX_NEIGHBOR_PAIRS = 25_000_000
SILHOUETTE_SAMPLE = 4_000
MAX_LEGEND_ITEMS = 40

NOISE = "Noise (-1)"
OTHER = "Other clusters"
RESERVED_PLOT_COLUMNS = {"Latitude", "Longitude", "Cluster", "Membership probability", "x", "y", "z"}

METRICS = {
    "euclidean": "Euclidean (as in the notebook)",
    "manhattan": "Manhattan (city block)",
    "chebyshev": "Chebyshev (max coordinate difference)",
    "haversine": "Haversine (great-circle distance, km)",
}
SCALINGS = {
    "none": "None (raw values)",
    "standard": "Standardize (z-score)",
    "minmax": "Min-max to [0, 1]",
    "robust": "Robust (median / IQR)",
}
SCALER_CLASSES = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
BACKENDS = {"hdbscan": "hdbscan library (as in the notebook)", "sklearn": "scikit-learn"}
MAP_STYLES = {
    "carto-positron": "Light (CARTO Positron)",
    "open-street-map": "OpenStreetMap",
    "carto-darkmatter": "Dark (CARTO Dark Matter)",
    "white-bg": "No basemap",
}
DARK_MAP_STYLES = {"carto-darkmatter"}
ONLINE_BASEMAPS = {
    "Online: CARTO Positron": ("CartoDB", "Positron"),
    "Online: OpenStreetMap": ("OpenStreetMap", "Mapnik"),
    "Online: Esri World Topo": ("Esri", "WorldTopoMap"),
}
VIEW_LABELS = {
    "map": "Map (latitude / longitude)",
    "pair": "Two features",
    "pca2": "PCA, 2 components",
    "pca3": "PCA, 3 components",
    "strip": "1-D strip plot",
}

# Validated categorical palette (8 hues, fixed order) and chart ink, light and dark.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "surface": "#fcfcfb"},
    "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781", "surface": "#1a1a19"},
}

NOTEBOOK_NOTES = {
    "missing": (
        "In the notebook (Implementation 1) missing values in ODCAF are strings of two dots `'..'`. "
        "The app reads every marker listed under **Read options** in the sidebar as missing, so such "
        "columns are recognised as numeric right away."
    ),
    "scaling_geo": (
        "**Notebook rationale.** DBSCAN is sensitive to scale. For coordinates, standardization would be "
        "an error because the data do not use the full latitude/longitude range. Latitude spans ±90° and "
        "longitude ±180°, so the notebook doubles the latitude (`coords_scaled['Latitude'] = 2 * ...`) "
        "and uses Euclidean distance. The haversine metric is the exact alternative: real distances on the "
        "Earth's surface, with eps in kilometres."
    ),
    "dbscan": (
        "DBSCAN labels a point as a **core point** when at least `min_samples` points (itself included) lie "
        "within distance `eps`. Clusters are chains of core points plus the border points they reach; "
        "everything else is **noise** (label -1). A single global `eps` cannot fit regions of different "
        "density at once: dense areas connected by chains of points merge into very large clusters. In the "
        "notebook, one cluster held 701 of the 1,607 museums."
    ),
    "hdbscan": (
        "HDBSCAN runs DBSCAN-like clusterings over all values of eps, builds the cluster hierarchy and keeps "
        "the most stable clusters, so it adapts to regions of different density and needs no `eps`. "
        "`min_cluster_size` is the smallest group that counts as a cluster; `min_samples` (None means equal "
        "to `min_cluster_size`) sets how conservative the result is: larger values give more noise. In the "
        "notebook, HDBSCAN found about 140 clusters of fairly uniform size (and more noise)."
    ),
    "compare": (
        "The notebook's closing questions: *What is different compared to DBSCAN? It might seem like there are "
        "more points identified as noise, but is that the case? Can we see the variations in density that "
        "HDBSCAN captures?* The tables below answer them for the current data."
    ),
}


# =============================================================================
# Small utilities
# =============================================================================
def _fill_width(func) -> dict:
    """Keyword arguments that make a Streamlit element use the full width, on old and new versions."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


FILL_PLOTLY = _fill_width(st.plotly_chart)
FILL_DF = _fill_width(st.dataframe)
FILL_IMAGE = _fill_width(st.image)


def _hash(*parts) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(repr(part).encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()[:10]


def _set_state(key: str, value) -> None:
    st.session_state[key] = value


def stateful(widget, label: str, key: str, default, **kwargs):
    """Create a widget whose default is set once through session state, so it never resets when the data
    behind the default change (e.g. after filtering)."""
    if key not in st.session_state:
        st.session_state[key] = default
    return widget(label, key=key, **kwargs)


def py_value(value):
    """numpy scalar -> plain Python value (for widgets and generated code)."""
    return value.item() if isinstance(value, np.generic) else value


def fmt_num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:.{digits}g}" if abs(value) < 1000 else f"{value:,.0f}"


def nice_number(value: float) -> float:
    """Round to 2 significant digits (0.8071 -> 0.81, 51.56 -> 52)."""
    if not value or not math.isfinite(value):
        return 1.0
    return float(f"{value:.2g}")


def step_for(scale: float) -> float:
    scale = abs(scale) if scale else 1.0
    return float(10 ** (math.floor(math.log10(scale)) - 1))


def format_for(scale: float) -> str:
    scale = abs(scale) if scale else 1.0
    decimals = min(10, max(0, -math.floor(math.log10(scale)) + 2))
    return f"%.{decimals}f"


def theme_type() -> str:
    try:
        kind = st.context.theme.type
    except Exception:
        kind = None
    return kind if kind in ("light", "dark") else "light"


# =============================================================================
# Loading CSV files
# =============================================================================
def detect_container(raw: bytes) -> str:
    if raw[:4] == b"PK\x03\x04":
        return "zip"
    if raw[:2] == b"\x1f\x8b":
        return "gzip"
    return "plain"


def zip_members(raw: bytes) -> list[tuple[str, int]]:
    """Tabular files inside a zip archive, largest first."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [
            (info.filename, info.file_size)
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(TABULAR_EXTENSIONS)
            and "__MACOSX" not in info.filename
            and not Path(info.filename).name.startswith(".")
        ]
    return sorted(members, key=lambda member: -member[1])


@st.cache_data(show_spinner=False, max_entries=4)
def unpack(raw: bytes, container: str, member: str | None) -> bytes:
    if container == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return archive.read(member)
    if container == "gzip":
        return gzip.decompress(raw)
    return raw


def detect_encoding(raw: bytes) -> str:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    for encoding in ("utf-8", "cp1252"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(text: str) -> str:
    """Delimiter giving a header and rows with the same, consistent number of fields."""
    lines = [line for line in text.splitlines() if line.strip()][:200]
    if not lines:
        return ","
    best, best_score = ",", None
    for delimiter in (",", ";", "\t", "|"):
        try:
            rows = list(csv.reader(lines, delimiter=delimiter))
        except csv.Error:
            continue
        counts = [len(row) for row in rows]
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        score = (counts[0] == mode, counts.count(mode) / len(counts), mode)
        if best_score is None or score > best_score:
            best, best_score = delimiter, score
    return best


def detect_decimal(text: str, sep: str) -> str:
    if sep == ",":
        return "."
    comma = len(re.findall(r"\d,\d", text))
    dot = len(re.findall(r"\d\.\d", text))
    return "," if comma > dot else "."


NUMBER_RE = re.compile(r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?")


def detect_header(text: str, sep: str) -> bool:
    """False when every field of the first line is a number (a file without column names)."""
    first = next((line for line in text.splitlines() if line.strip()), "")
    if sep == r"\s+":
        fields = first.split()
    else:
        fields = next(csv.reader([first], delimiter=sep), [])
    fields = [f.strip().strip('"') for f in fields if f.strip()]
    return not fields or not all(NUMBER_RE.fullmatch(f) for f in fields)


@st.cache_data(show_spinner=False, max_entries=8)
def detect_format(raw: bytes) -> dict:
    encoding = detect_encoding(raw)
    sample = raw[:200_000].decode(encoding, errors="replace")
    sep = detect_delimiter(sample)
    return {
        "encoding": encoding,
        "sep": sep,
        "decimal": detect_decimal(sample, sep),
        "header": detect_header(sample, sep),
    }


def read_csv_kwargs(read: dict) -> dict:
    kwargs = {
        "encoding": read["encoding"],
        "sep": read["sep"],
        "decimal": read["decimal"],
        "header": 0 if read["header"] else None,
        "na_values": list(read["na_values"]),
        "keep_default_na": True,
        "skipinitialspace": True,
        "low_memory": False,
    }
    if read.get("skip_bad_lines"):
        kwargs["on_bad_lines"] = "skip"
    return kwargs


@st.cache_data(show_spinner="Reading the CSV file…", max_entries=6)
def parse_csv(raw: bytes, encoding: str, sep: str, decimal: str, header: bool, na_values: tuple) -> tuple:
    """Returns (DataFrame, skipped_bad_lines)."""
    read = {"encoding": encoding, "sep": sep, "decimal": decimal, "header": header, "na_values": na_values}
    try:
        df = pd.read_csv(io.BytesIO(raw), **read_csv_kwargs(read))
        skipped = False
    except pd.errors.ParserError:
        df = pd.read_csv(io.BytesIO(raw), **read_csv_kwargs({**read, "skip_bad_lines": True}))
        skipped = True
    if not header:
        df.columns = [f"col_{i + 1}" for i in range(df.shape[1])]
    return df, skipped


@st.cache_data(show_spinner=False, ttl=30)
def list_data_files(folder: str) -> list:
    root = Path(folder)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.rglob("*")):
        name = path.name.lower()
        if not path.is_file() or name.startswith("."):
            continue
        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(path) as archive:
                    if not any(m.lower().endswith(TABULAR_EXTENSIONS) for m in archive.namelist()):
                        continue
            except (zipfile.BadZipFile, OSError):
                continue
        elif not name.endswith(FOLDER_EXTENSIONS):
            continue
        found.append(path.relative_to(root).as_posix())
    return found


@st.cache_data(show_spinner=False, ttl=30)
def list_geotiffs(folder: str) -> list:
    root = Path(folder)
    if not root.is_dir():
        return []
    return [p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.suffix.lower() in (".tif", ".tiff")]


def default_file_index(files: list) -> int:
    for i, name in enumerate(files):
        if re.search(r"odcaf_v1[-.]0\.csv$", name.lower()):
            return i
    return 0


def normalise_url(url: str) -> str:
    url = url.strip()
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if match:
        return f"https://raw.githubusercontent.com/{match.group(1)}/{match.group(2)}/{match.group(3)}"
    return url


@st.cache_data(show_spinner="Downloading…", ttl=3600, max_entries=4)
def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (DBSCAN-HDBSCAN explorer)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def is_notebook_dataset(df: pd.DataFrame) -> bool:
    return {"ODCAF_Facility_Type", "Latitude", "Longitude"}.issubset(df.columns)


# =============================================================================
# Column typing helpers
# =============================================================================
def to_numeric_lenient(s: pd.Series) -> pd.Series:
    """Numbers stored as text (e.g. '49.89,') -> float; anything else -> NaN."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    text = s.astype("string").str.strip().str.rstrip(",;").str.strip()
    return pd.to_numeric(text, errors="coerce").astype(float)


@st.cache_data(show_spinner=False, max_entries=8)
def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        numeric_count = int(to_numeric_lenient(s).notna().sum())
        first = s.dropna()
        rows.append(
            {
                "column": str(col),
                "dtype": str(s.dtype),
                "non-null": non_null,
                "missing %": round(100 * (1 - non_null / n), 1) if n else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "numeric %": round(100 * numeric_count / non_null, 1) if non_null else 0.0,
                "numeric values": numeric_count,
                "example": str(first.iloc[0])[:60] if len(first) else "",
            }
        )
    profile = pd.DataFrame(rows)
    profile.index = profile["column"].tolist()
    return profile


def numeric_like_columns(profile: pd.DataFrame) -> list:
    ok = (profile["numeric %"] >= 90) & (profile["numeric values"] >= 2)
    return profile.loc[ok, "column"].tolist()


def looks_like_id(name: str, values: pd.Series) -> bool:
    low = name.strip().lower()
    if low in {"id", "index", "idx", "#", "no", "nr", "row", "rowid", "key"} or low.startswith("unnamed"):
        return True
    if re.search(r"(^|[^a-z])(id|uid|code|key)$", low) or name.endswith(("ID", "Id", "UID")):
        return True
    v = to_numeric_lenient(values).dropna()
    return bool(len(v) > 20 and v.is_unique and v.is_monotonic_increasing and np.allclose(v, np.round(v)))


def _name_tokens(name: str) -> set:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _in_range(values: pd.Series, limit: float) -> bool:
    v = to_numeric_lenient(values).dropna()
    return bool(len(v) and (v.abs() <= limit).mean() >= 0.98)


def guess_lat_lon(df: pd.DataFrame, numeric_cols: list) -> tuple:
    lat = lon = None
    for col in numeric_cols:
        tokens = _name_tokens(col)
        if lat is None and (tokens & {"lat", "latitude", "lati"} or "latitude" in col.lower()):
            if _in_range(df[col], 90):
                lat = col
        elif lon is None and (tokens & {"lon", "lng", "long", "longitude", "longi"} or "longitude" in col.lower()):
            if _in_range(df[col], 180):
                lon = col
    return lat, lon


def coordinate_problem(df: pd.DataFrame, lat: str, lon: str) -> str | None:
    if lat == lon:
        return "Latitude and longitude must be two different columns."
    if not _in_range(df[lat], 90):
        return f"'{lat}' has values outside ±90°, so it is not a latitude in degrees."
    if not _in_range(df[lon], 180):
        return f"'{lon}' has values outside ±180°, so it is not a longitude in degrees."
    return None


def guess_hover_columns(columns: list) -> list:
    """One name-like and one place-like column, e.g. Facility_Name and City in ODCAF."""
    picks = []
    for words in ({"name", "title", "label"}, {"city", "town", "place", "country", "region", "province", "state"}):
        match = next((c for c in columns if _name_tokens(c) & words and c not in picks), None)
        if match is not None:
            picks.append(match)
    return picks


# =============================================================================
# Prepared feature matrix
# =============================================================================
@dataclass
class Loaded:
    df: pd.DataFrame
    key: str
    name: str
    preset: bool
    source: dict


@dataclass
class Prepared:
    data: pd.DataFrame          # rows that are clustered, all original columns
    X: np.ndarray               # model input (the notebook's coords_scaled)
    model_columns: list
    features: list
    categorical: list
    metric: str
    scaling: str
    lat_col: str | None
    lon_col: str | None
    space_key: str
    is_notebook: bool
    recipe: dict

    @property
    def has_geo(self) -> bool:
        return self.lat_col is not None and self.lon_col is not None

    @property
    def unit(self) -> str:
        return "km" if self.metric == "haversine" else "model units"

    def to_model_distance(self, value: float) -> float:
        return value / EARTH_RADIUS_KM if self.metric == "haversine" else value


def build_matrix(clean, features, categorical, metric, scaling, weights, lat_col, lon_col) -> tuple:
    """Model input exactly as written in the exported script."""
    if metric == "haversine":
        return np.radians(clean[[lat_col, lon_col]].to_numpy(dtype=float)), [lat_col, lon_col]
    frame = clean[features + categorical]
    if categorical:
        frame = pd.get_dummies(frame, columns=categorical, dtype=float)
    columns = [str(c) for c in frame.columns]
    X = frame.to_numpy(dtype=float)
    if scaling in SCALER_CLASSES:
        X = SCALER_CLASSES[scaling]().fit_transform(X)
    w = np.array([weights.get(c, 1.0) for c in columns], dtype=float)
    return X * w, columns


# =============================================================================
# Clustering (cached)
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def k_distances(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Sorted distance of every point to its k-th nearest neighbour (itself included, as min_samples)."""
    k = int(min(max(k, 2), len(X)))
    dist = NearestNeighbors(n_neighbors=k, metric=metric).fit(X).kneighbors(X)[0][:, -1]
    if metric == "haversine":
        dist = dist * EARTH_RADIUS_KM
    return np.sort(dist)


def suggest_eps(kd: np.ndarray) -> float:
    """Knee of the sorted k-distance curve: the point farthest below the chord joining its ends.

    Extreme outliers are capped at 3x the 99th percentile so that a few isolated points cannot flatten the
    curve. On the notebook's museums (min_samples = 3) this gives eps = 1.0, the value the notebook uses.
    """
    y = np.sort(np.asarray(kd, dtype=float))
    y = y[np.isfinite(y)]
    positive = y[y > 0]
    fallback = float(positive.min()) if positive.size else 1.0
    if y.size < 3:
        return float(y.max()) if y.size and y.max() > 0 else fallback
    y = np.minimum(y, 3 * np.percentile(y, 99))
    span = y[-1] - y[0]
    if span <= 0:
        return fallback
    x = np.linspace(0.0, 1.0, y.size)
    value = float(y[int(np.argmax(x - (y - y[0]) / span))])
    return value if value > 0 else fallback


@st.cache_data(show_spinner=False, max_entries=32)
def estimate_neighbor_pairs(X: np.ndarray, eps: float, metric: str) -> float:
    """Rough number of eps-neighbour pairs DBSCAN must hold in memory."""
    n = len(X)
    if n <= 5000:
        return 0.0
    idx = np.random.default_rng(0).choice(n, size=300, replace=False)
    model = NearestNeighbors(radius=eps, metric=metric).fit(X)
    counts = [len(a) for a in model.radius_neighbors(X[idx], return_distance=False)]
    return float(np.mean(counts) * n)


@st.cache_data(show_spinner=False, max_entries=24)
def run_dbscan(X: np.ndarray, eps: float, min_samples: int, metric: str) -> tuple:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    core = np.zeros(len(labels), dtype=bool)
    core[model.core_sample_indices_] = True
    return labels, core


def sklearn_min_samples(min_samples, min_cluster_size) -> int:
    """scikit-learn counts the point itself in min_samples, the hdbscan library does not."""
    return int(min_samples if min_samples is not None else min_cluster_size) + 1


@st.cache_data(show_spinner=False, max_entries=24)
def run_hdbscan(X, backend, min_cluster_size, min_samples, metric, selection_epsilon, method, single) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if backend == "hdbscan":
            model = hdbscan_lib.HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=None if min_samples is None else int(min_samples),
                metric=metric,
                cluster_selection_epsilon=float(selection_epsilon),
                cluster_selection_method=method,
                allow_single_cluster=bool(single),
            )
        else:
            model = SkHDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=sklearn_min_samples(min_samples, min_cluster_size),
                metric=metric,
                cluster_selection_epsilon=float(selection_epsilon),
                cluster_selection_method=method,
                allow_single_cluster=bool(single),
                copy=True,
            )
        labels = model.fit_predict(X)
    return np.asarray(labels), np.asarray(model.probabilities_, dtype=float)


def summarize(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n = len(labels)
    noise = int((labels == -1).sum())
    sizes = pd.Series(labels[labels != -1]).value_counts()
    has = len(sizes) > 0
    return {
        "points": n,
        "clusters": int(len(sizes)),
        "noise": noise,
        "noise_pct": 100.0 * noise / n if n else 0.0,
        "largest": int(sizes.iloc[0]) if has else 0,
        "largest_pct": 100.0 * sizes.iloc[0] / n if has and n else 0.0,
        "median": float(sizes.median()) if has else 0.0,
        "smallest": int(sizes.iloc[-1]) if has else 0,
    }


@st.cache_data(show_spinner=False, max_entries=32)
def cluster_quality(X: np.ndarray, labels: np.ndarray, metric: str) -> dict:
    out = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    mask = labels != -1
    k = len(np.unique(labels[mask]))
    if k < 2 or mask.sum() <= k:
        return out
    Xc, lc = X[mask], labels[mask]
    sample = SILHOUETTE_SAMPLE if len(Xc) > SILHOUETTE_SAMPLE else None
    try:
        out["silhouette"] = float(silhouette_score(Xc, lc, metric=metric, sample_size=sample, random_state=0))
    except Exception:
        pass
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
    except Exception:
        pass
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xc, lc))
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    out = {"ari": float(adjusted_rand_score(a, b)), "ami": None}
    if len(a) <= 30_000:
        out["ami"] = float(adjusted_mutual_info_score(a, b))
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def pca_projection(X: np.ndarray, n_components: int) -> tuple:
    n_components = min(n_components, X.shape[1], len(X))
    model = PCA(n_components=n_components, random_state=0)
    coords = model.fit_transform(X)
    return coords, model.explained_variance_ratio_


# =============================================================================
# Figures
# =============================================================================
def cluster_groups(labels: np.ndarray, coloring: str, palette: list, noise_color: str, other_color: str) -> tuple:
    """Group names per point, colour per group and drawing order (noise first, so it sits underneath)."""
    labels = np.asarray(labels)
    sizes = pd.Series(labels[labels != -1]).value_counts()
    if coloring == "top" and len(sizes) > len(palette):
        top = [int(c) for c in sizes.index[: len(palette)]]
        names = np.where(labels == -1, NOISE, np.where(np.isin(labels, top), labels.astype(str), OTHER))
        colors = {str(c): palette[i] for i, c in enumerate(top)}
        colors.update({OTHER: other_color, NOISE: noise_color})
        order = [NOISE, OTHER] + [str(c) for c in reversed(top)]
    else:
        ids = sorted(int(c) for c in sizes.index)
        names = np.where(labels == -1, NOISE, labels.astype(str))
        colors = {str(c): palette[c % len(palette)] for c in ids}
        colors[NOISE] = noise_color
        order = [NOISE] + [str(c) for c in reversed(ids)]
    return names, colors, order


def map_view(lat: np.ndarray, lon: np.ndarray, height: int, width: int = 800) -> tuple:
    """Center and zoom that fit all points (Web Mercator, 512 px tiles)."""
    lat = np.clip(lat, -85.0, 85.0)

    def merc(v):
        return np.log(np.tan(np.pi / 4 + np.radians(v) / 2))

    y_min, y_max = merc(lat.min()), merc(lat.max())
    x_span = max(np.radians(lon.max() - lon.min()), 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    zoom = min(np.log2(width * 2 * np.pi / (512 * x_span)), np.log2(height * 2 * np.pi / (512 * y_span)))
    center_lat = np.degrees(2 * np.arctan(np.exp((y_min + y_max) / 2)) - np.pi / 2)
    center = {"lat": float(center_lat), "lon": float((lon.min() + lon.max()) / 2)}
    return center, float(np.clip(zoom - 0.2, 0.0, 15.0))


def _hover_frame(prep: Prepared, view: dict) -> pd.DataFrame | None:
    cols = [c for c in view.get("hover", []) if c in prep.data.columns and c not in RESERVED_PLOT_COLUMNS]
    if not cols:
        return None
    return prep.data[cols].reset_index(drop=True)


def _finish(fig: go.Figure, n_groups: int, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(title_text="Cluster", traceorder="reversed", itemsizing="constant"),
        showlegend=n_groups <= MAX_LEGEND_ITEMS,
    )
    return fig


def map_figure(prep, groups, colors, order, view, title, extra, height, width) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Latitude": prep.data[prep.lat_col].to_numpy(dtype=float),
            "Longitude": prep.data[prep.lon_col].to_numpy(dtype=float),
            "Cluster": groups,
        }
    )
    hover = _hover_frame(prep, view)
    hover_data = {"Latitude": ":.4f", "Longitude": ":.4f"}
    if hover is not None:
        for c in hover.columns:
            frame[c] = hover[c].to_numpy()
            hover_data[c] = True
    for name, values in extra.items():
        frame[name] = values
        hover_data[name] = ":.2f"
    frame = frame.dropna(subset=["Latitude", "Longitude"])
    if frame.empty:
        return go.Figure().update_layout(title="No valid coordinates to plot", height=height)
    legend_px = 90 if len(order) <= MAX_LEGEND_ITEMS else 0
    center, zoom = map_view(frame["Latitude"].to_numpy(), frame["Longitude"].to_numpy(), height - 60, width - legend_px)
    common = dict(
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        color_discrete_map=colors,
        category_orders={"Cluster": order},
        hover_data=hover_data,
        zoom=zoom,
        center=center,
        title=title,
    )
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(frame, map_style=view["map_style"], **common)
    else:  # plotly < 5.24
        fig = px.scatter_mapbox(frame, mapbox_style=view["map_style"], **common)
    size = view["point_size"]
    fig.update_traces(marker=dict(size=size, opacity=0.85))
    fig.for_each_trace(
        lambda t: t.update(marker=dict(size=max(3, size - 2), opacity=0.9)) if t.name in (NOISE, OTHER) else None
    )
    return _finish(fig, len(order), height)


def projection(prep: Prepared, view: dict) -> tuple:
    kind = view["kind"]
    if kind == "pair":
        fx, fy = view["x"], view["y"]
        coords = prep.data[[fx, fy]].to_numpy(dtype=float)
        return coords, [fx, fy]
    if kind in ("pca2", "pca3"):
        coords, ratio = pca_projection(prep.X, 2 if kind == "pca2" else 3)
        axes = [f"PC{i + 1} ({100 * r:.0f}% of variance)" for i, r in enumerate(ratio)]
        return coords, axes
    jitter = np.random.default_rng(0).uniform(-1, 1, len(prep.X))
    return np.column_stack([prep.X[:, 0], jitter]), [prep.model_columns[0] + " (model space)", "jitter"]


def space_figure(prep, groups, colors, order, view, title, extra, height) -> go.Figure:
    coords, axes = projection(prep, view)
    names = ["x", "y", "z"][: coords.shape[1]]
    frame = pd.DataFrame(coords, columns=names)
    frame["Cluster"] = groups
    hover = _hover_frame(prep, view)
    hover_data = {}
    if hover is not None:
        for c in hover.columns:
            frame[c] = hover[c].to_numpy()
            hover_data[c] = True
    for name, values in extra.items():
        frame[name] = values
        hover_data[name] = ":.2f"
    common = dict(
        color="Cluster",
        color_discrete_map=colors,
        category_orders={"Cluster": order},
        hover_data=hover_data,
        labels=dict(zip(names, axes)),
        title=title,
    )
    surface = INK[view["theme"]]["surface"]
    if len(names) == 3:
        fig = px.scatter_3d(frame, x="x", y="y", z="z", **common)
        fig.update_traces(marker=dict(size=max(2, view["point_size"] // 2), opacity=0.85))
    else:
        fig = px.scatter(frame, x="x", y="y", render_mode="webgl", **common)
        fig.update_traces(marker=dict(size=view["point_size"], opacity=0.85, line=dict(width=1, color=surface)))
    return _finish(fig, len(order), height)


def cluster_figure(prep: Prepared, labels: np.ndarray, view: dict, title: str, probabilities=None, height=600, width=950):
    if view["kind"] == "map":
        mode = "dark" if view["map_style"] in DARK_MAP_STYLES else "light"
    else:
        mode = view["theme"]
    groups, colors, order = cluster_groups(
        labels, view["coloring"], SERIES[mode], INK[mode]["primary"], INK[mode]["muted"]
    )
    extra = {} if probabilities is None else {"Membership probability": np.asarray(probabilities)}
    if view["kind"] == "map":
        return map_figure(prep, groups, colors, order, view, title, extra, height, width), len(order)
    return space_figure(prep, groups, colors, order, view, title, extra, height), len(order)


def kdistance_figure(kd: np.ndarray, eps: float, suggestion: float, k: int, unit: str, theme: str) -> go.Figure:
    n = len(kd)
    idx = np.unique(np.r_[np.arange(0, n, max(1, n // 3000)), n - 1])
    series, ink = SERIES[theme], INK[theme]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=kd[idx],
            mode="lines",
            line=dict(color=series[0], width=2),
            name=f"distance to neighbour #{k}",
            hovertemplate="rank %{x}<br>distance %{y:.4g}<extra></extra>",
        )
    )
    knee_x = int(min(np.searchsorted(kd, suggestion), n - 1))
    fig.add_trace(
        go.Scatter(
            x=[knee_x],
            y=[suggestion],
            mode="markers",
            marker=dict(size=11, color=series[1], line=dict(width=2, color=ink["surface"])),
            name=f"suggested eps ≈ {fmt_num(suggestion)}",
            hovertemplate="suggested eps %{y:.4g}<extra></extra>",
        )
    )
    fig.add_hline(
        y=eps,
        line=dict(color=ink["secondary"], width=1, dash="dash"),
        annotation_text=f"current eps = {fmt_num(eps)}",
        annotation_position="top left",
    )
    top = max(float(np.percentile(kd, 99.5)), eps, suggestion) * 1.15
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="points sorted by distance",
        yaxis_title=f"distance ({unit})",
        yaxis_range=[0, top],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def size_bar_figure(labels: np.ndarray, theme: str, top: int = 40) -> go.Figure:
    counts = pd.Series(labels).value_counts()
    clusters = counts.drop(-1, errors="ignore").head(top)
    names = [str(c) for c in clusters.index]
    values = clusters.tolist()
    colors = [SERIES[theme][0]] * len(names)
    if -1 in counts.index:
        names.append(NOISE)
        values.append(int(counts[-1]))
        colors.append(INK[theme]["muted"])
    fig = go.Figure(
        go.Bar(x=names, y=values, marker_color=colors, hovertemplate="cluster %{x}<br>%{y} points<extra></extra>")
    )
    shown = f"largest {top} clusters" if len(counts.drop(-1, errors="ignore")) > top else "clusters"
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(type="category", title=f"{shown}, largest first, then noise"),
        yaxis_title="points",
        bargap=0.25,
        showlegend=False,
    )
    return fig


def size_distribution_figure(results: dict, theme: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, labels) in enumerate(results.items()):
        sizes = pd.Series(labels[labels != -1]).value_counts().to_numpy()
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(sizes) + 1),
                y=sizes,
                mode="lines+markers",
                name=name,
                line=dict(width=2, color=SERIES[theme][i]),
                marker=dict(size=6),
                hovertemplate=name + ": cluster rank %{x}<br>%{y} points<extra></extra>",
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="cluster rank (1 = largest)",
        yaxis_title="points in the cluster (log scale)",
        yaxis=dict(type="log", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def probability_figure(probabilities: np.ndarray, labels: np.ndarray, theme: str) -> go.Figure:
    values = probabilities[labels != -1]
    fig = go.Figure(go.Histogram(x=values, nbinsx=20, marker_color=SERIES[theme][0]))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="membership probability (clustered points)",
        yaxis_title="points",
        bargap=0.1,
    )
    return fig


def to_web_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple:
    lat = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = WEB_MERCATOR_RADIUS * np.radians(np.asarray(lon, dtype=float))
    y = WEB_MERCATOR_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


@st.cache_data(show_spinner="Drawing the static map…", max_entries=12, ttl=3600)
def static_map_png(lat: np.ndarray, lon: np.ndarray, labels: np.ndarray, title: str, basemap: str, area: float):
    """The notebook's plot_clustered_locations: Web Mercator, noise black/red, clusters tab10, basemap."""
    ok = np.isfinite(lat) & np.isfinite(lon)
    x, y = to_web_mercator(lon[ok], lat[ok])
    labels = labels[ok]
    fig, ax = plt.subplots(figsize=(15, 10))
    noise = labels == -1
    ax.scatter(x[noise], y[noise], color="k", s=area, edgecolors="r", alpha=1, label="Noise", zorder=2)
    if (~noise).any():
        tab10 = plt.get_cmap("tab10").colors
        colors = [tab10[int(c) % 10] for c in labels[~noise]]
        ax.scatter(x[~noise], y[~noise], c=colors, s=area, edgecolors="k", alpha=0.6, zorder=3)
    pad_x = max((x.max() - x.min()) * 0.05, 5_000) if len(x) else 1
    pad_y = max((y.max() - y.min()) * 0.05, 5_000) if len(y) else 1
    if len(x):
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
    warning = None
    if basemap != "none":
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        try:
            if ctx is None:
                raise RuntimeError("contextily is not installed (pip install contextily)")
            if basemap.startswith("file:"):
                ctx.add_basemap(ax, source=basemap[5:], crs="EPSG:3857", reset_extent=False)
                if ax.images and len(x):  # keep the padding inside the GeoTIFF, never hide a point
                    left, right, bottom, top = ax.images[-1].get_extent()
                    xlim = (max(xlim[0], left) if left <= x.min() else xlim[0],
                            min(xlim[1], right) if right >= x.max() else xlim[1])
                    ylim = (max(ylim[0], bottom) if bottom <= y.min() else ylim[0],
                            min(ylim[1], top) if top >= y.max() else ylim[1])
                    outside = int(((x < left) | (x > right) | (y < bottom) | (y > top)).sum())
                    if outside:
                        warning = (f"{outside:,} of {len(x):,} points lie outside this GeoTIFF; "
                                   "choose an online basemap for this dataset.")
            else:
                group, name = ONLINE_BASEMAPS[basemap[7:]]
                ctx.add_basemap(ax, source=ctx.providers[group][name], crs="EPSG:3857")
        except Exception as exc:
            warning = f"The basemap could not be drawn: {exc}"
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=80)
    plt.close(fig)
    return buffer.getvalue(), warning


# =============================================================================
# Tables
# =============================================================================
def size_table(labels: np.ndarray) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts()
    table = pd.DataFrame({"Cluster": counts.index, "Points": counts.to_numpy()})
    table["Share %"] = (100 * table["Points"] / len(labels)).round(1)
    table["Cluster"] = [NOISE if c == -1 else str(c) for c in table["Cluster"]]
    return table


def profile_table(prep: Prepared, labels: np.ndarray) -> pd.DataFrame:
    numeric = list(prep.features)
    if prep.has_geo:
        numeric += [c for c in (prep.lat_col, prep.lon_col) if c not in numeric]
    frame = prep.data[numeric + prep.categorical].copy()
    frame["Cluster"] = labels
    groups = frame.groupby("Cluster")
    table = groups[numeric].mean().round(4) if numeric else pd.DataFrame(index=groups.size().index)
    for c in prep.categorical:
        table[f"{c} (most common)"] = groups[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    table.insert(0, "Points", groups.size())
    table = table.sort_values("Points", ascending=False).reset_index()
    table["Cluster"] = [NOISE if c == -1 else str(c) for c in table["Cluster"]]
    return table


def split_table(a: np.ndarray, b: np.ndarray, top: int = 8) -> pd.DataFrame:
    sizes = pd.Series(a[a != -1]).value_counts().head(top)
    rows = []
    for cluster, size in sizes.items():
        inside = b[a == cluster]
        rows.append(
            {
                "DBSCAN cluster": str(cluster),
                "Points": int(size),
                "HDBSCAN clusters inside": int(len(np.unique(inside[inside != -1]))),
                "Marked noise by HDBSCAN %": round(100 * float((inside == -1).mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def noise_table(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    an, bn = a == -1, b == -1
    return pd.DataFrame(
        {
            "HDBSCAN: clustered": [int((~an & ~bn).sum()), int((an & ~bn).sum())],
            "HDBSCAN: noise": [int((~an & bn).sum()), int((an & bn).sum())],
        },
        index=["DBSCAN: clustered", "DBSCAN: noise"],
    )


# =============================================================================
# Exported script
# =============================================================================
def generate_script(r: dict) -> str:
    """A standalone Python script that reproduces the configured run (same steps as the notebook)."""
    src, read = r["source"], r["source"]["read"]
    haversine = r["metric"] == "haversine"
    container, kind = src["container"], src["kind"]
    db, hd = r.get("dbscan"), r.get("hdbscan")
    scaler = None if haversine else SCALER_CLASSES.get(r["scaling"])
    lines: list = []
    add = lines.append

    add('"""Clustering pipeline exported from the DBSCAN & HDBSCAN explorer (Streamlit app).')
    add("")
    add("Same steps as the 'Comparing DBSCAN and HDBSCAN clustering' notebook:")
    add("load -> filter rows -> select features -> clean -> scale -> DBSCAN -> HDBSCAN -> plot.")
    add('"""')
    if kind == "url" or container == "zip":
        add("import io")
    if kind == "url":
        add("import urllib.request")
    if container == "zip":
        add("import zipfile")
    add("")
    add("import numpy as np")
    add("import pandas as pd")
    add("from sklearn.cluster import DBSCAN")
    if hd and hd["backend"] == "hdbscan":
        add("import hdbscan")
    elif hd:
        add("from sklearn.cluster import HDBSCAN")
    if scaler is not None:
        add(f"from sklearn.preprocessing import {scaler.__name__}")
    add("")
    if haversine:
        add("EARTH_RADIUS_KM = 6371.0088")
        add("")
    add("")
    add("def to_numeric_lenient(s):")
    add('    """Numbers stored as text (e.g. \'49.89,\') -> float; anything else -> NaN."""')
    add("    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):")
    add("        return pd.to_numeric(s, errors='coerce').astype(float)")
    add("    text = s.astype('string').str.strip().str.rstrip(',;').str.strip()")
    add("    return pd.to_numeric(text, errors='coerce').astype(float)")
    add("")
    add("")
    add("# 1. Load the data")
    header_value = 0 if read["header"] else None
    add("READ_OPTIONS = dict(")
    add(f"    encoding={read['encoding']!r},")
    add(f"    sep={read['sep']!r},")
    add(f"    decimal={read['decimal']!r},")
    add(f"    header={header_value!r},")
    add(f"    na_values={list(read['na_values'])!r},  # extra markers read as missing values")
    add("    keep_default_na=True,")
    add("    skipinitialspace=True,")
    add("    low_memory=False,")
    if read.get("skip_bad_lines"):
        add("    on_bad_lines='skip',")
    add(")")
    if kind == "url":
        add(f"URL = {src['ref']!r}")
        add("request = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})")
        add("with urllib.request.urlopen(request) as response:")
        add("    raw = response.read()")
        if container == "zip":
            add("with zipfile.ZipFile(io.BytesIO(raw)) as archive:")
            add(f"    with archive.open({src['member']!r}) as f:")
            add("        df = pd.read_csv(f, **READ_OPTIONS)")
        elif container == "gzip":
            add("df = pd.read_csv(io.BytesIO(raw), compression='gzip', **READ_OPTIONS)")
        else:
            add("df = pd.read_csv(io.BytesIO(raw), **READ_OPTIONS)")
    else:
        note = "  # the file uploaded to the app: adjust the path" if kind == "upload" else ""
        add(f"PATH = {src['ref']!r}{note}")
        if container == "zip":
            add("with zipfile.ZipFile(PATH) as archive:")
            add(f"    with archive.open({src['member']!r}) as f:")
            add("        df = pd.read_csv(f, **READ_OPTIONS)")
        elif container == "gzip":
            add("df = pd.read_csv(PATH, compression='gzip', **READ_OPTIONS)")
        else:
            add("df = pd.read_csv(PATH, **READ_OPTIONS)")
    if not read["header"]:
        add("df.columns = [f'col_{i + 1}' for i in range(df.shape[1])]")
    add("")
    add("# 2. Filter the rows (the notebook keeps the museums)")
    if not r["filters"]:
        add("# no filter: every row is kept")
    for flt in r["filters"]:
        col = flt["column"]
        if flt["kind"] == "values":
            add(f"df = df[df[{col!r}].isin({flt['values']!r})]")
        else:
            add(f"df = df[to_numeric_lenient(df[{col!r}]).between({flt['min']!r}, {flt['max']!r})]")
    add("")
    add("# 3. Select the features, convert them to numbers and drop incomplete rows")
    add(f"features = {r['features']!r}")
    if r["categorical"]:
        add(f"categorical_features = {r['categorical']!r}  # one-hot encoded")
    add("df = df.copy()")
    convert = list(r["features"])
    if r["lat_col"]:
        convert += [c for c in (r["lat_col"], r["lon_col"]) if c not in convert]
    add(f"for col in {convert!r}:")
    add("    df[col] = to_numeric_lenient(df[col])")
    required = "features + categorical_features" if r["categorical"] else "features"
    add(f"df = df.dropna(subset={required})")
    if r.get("sample_n"):
        add(f"df = df.sample(n={r['sample_n']}, random_state=0).sort_index()  # random sample, as in the app")
    add("print(len(df), 'rows to cluster')")
    add("")
    add("# 4. Build the model input (the notebook's coords_scaled)")
    if haversine:
        add(f"X = np.radians(df[[{r['lat_col']!r}, {r['lon_col']!r}]].to_numpy(dtype=float))  # [lat, lon] in radians")
    else:
        if r["categorical"]:
            add("X = pd.get_dummies(df[features + categorical_features], columns=categorical_features, dtype=float)")
        else:
            add("X = df[features].copy()")
        add("X = X.astype(float)")
        if scaler is not None:
            add(f"X = pd.DataFrame({scaler.__name__}().fit_transform(X), columns=X.columns, index=X.index)")
        for col, weight in r["weights"].items():
            add(f"X[{col!r}] = {weight!r} * X[{col!r}]  # feature weight")
        add("X = X.to_numpy(dtype=float, copy=True)  # plain writable array (the hdbscan library needs one)")
    add("")
    if db:
        add("# 5. DBSCAN")
        add(f"min_samples = {int(db['min_samples'])}  # minimum number of samples needed to form a neighbourhood")
        if haversine:
            add(f"eps = {db['eps']!r} / EARTH_RADIUS_KM  # neighbourhood search radius ({db['eps']:g} km in radians)")
        else:
            add(f"eps = {db['eps']!r}  # neighbourhood search radius")
        add(f"metric = {r['metric']!r}  # distance measure")
        add("dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)")
        add("df['DBSCAN_Cluster'] = dbscan.fit_predict(X)")
        add("print(df['DBSCAN_Cluster'].value_counts())")
        add("")
    if hd:
        add("# 6. HDBSCAN")
        cse = float(hd["cluster_selection_epsilon"])
        cse_expr = f"{cse!r} / EARTH_RADIUS_KM" if haversine and cse > 0 else repr(cse)
        options = []
        if cse > 0:
            options.append(f"cluster_selection_epsilon={cse_expr}")
        if hd["cluster_selection_method"] != "eom":
            options.append(f"cluster_selection_method={hd['cluster_selection_method']!r}")
        if hd["allow_single_cluster"]:
            options.append("allow_single_cluster=True")
        if hd["backend"] == "hdbscan":
            args = [f"min_samples={hd['min_samples']!r}", f"min_cluster_size={int(hd['min_cluster_size'])}",
                    f"metric={r['metric']!r}"] + options
            add(f"hdb = hdbscan.HDBSCAN({', '.join(args)})")
        else:
            effective = sklearn_min_samples(hd["min_samples"], hd["min_cluster_size"])
            add("# scikit-learn counts the point itself in min_samples: notebook (hdbscan library) value + 1")
            args = [f"min_cluster_size={int(hd['min_cluster_size'])}", f"min_samples={effective}",
                    f"metric={r['metric']!r}"] + options + ["copy=True"]
            add(f"hdb = HDBSCAN({', '.join(args)})")
        add("df['HDBSCAN_Cluster'] = hdb.fit_predict(X)")
        add("df['HDBSCAN_Probability'] = hdb.probabilities_")
        add("print(df['HDBSCAN_Cluster'].value_counts())")
        add("")
    add("df.to_csv('clustered_data.csv', index_label='source_row')")
    if r["lat_col"] and (db or hd):
        basemap = r.get("static_basemap", "none")
        if basemap.startswith("file:"):
            basemap_expr = repr(basemap[5:])
        elif basemap.startswith("online:"):
            group, name = ONLINE_BASEMAPS[basemap[7:]]
            basemap_expr = f"ctx.providers.{group}.{name} if ctx else None"
        else:
            basemap_expr = "None"
        titles = (
            ("Museums Clustered by Proximity", "Museums Hierarchically Clustered by Proximity")
            if r.get("is_notebook")
            else ("DBSCAN clusters", "HDBSCAN clusters")
        )
        add("")
        add("")
        add("# 7. Plot the clusters on a basemap (the notebook's plot_clustered_locations)")
        add("import matplotlib.pyplot as plt")
        add("")
        add("try:")
        add("    import contextily as ctx  # basemaps; optional")
        add("except ImportError:")
        add("    ctx = None")
        add("")
        add("")
        add("def plot_clustered_locations(df, cluster_col, title, basemap=None):")
        add(f"    lat = df[{r['lat_col']!r}].to_numpy(dtype=float)")
        add(f"    lon = df[{r['lon_col']!r}].to_numpy(dtype=float)")
        add("    ok = np.isfinite(lat) & np.isfinite(lon)")
        add("    x = 6378137.0 * np.radians(lon[ok])  # Web Mercator (EPSG:3857), as gdf.to_crs(epsg=3857)")
        add("    y = 6378137.0 * np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat[ok], -85.05, 85.05)) / 2))")
        add("    labels = df[cluster_col].to_numpy()[ok]")
        add("    fig, ax = plt.subplots(figsize=(15, 10))")
        add("    noise = labels == -1")
        add("    ax.scatter(x[noise], y[noise], color='k', s=30, edgecolors='r', alpha=1, label='Noise')")
        add("    tab10 = plt.get_cmap('tab10').colors")
        add("    ax.scatter(x[~noise], y[~noise], c=[tab10[c % 10] for c in labels[~noise]], s=30, edgecolors='k', alpha=0.6)")
        add("    if ctx is not None and basemap is not None:")
        add("        xlim, ylim = ax.get_xlim(), ax.get_ylim()")
        add("        try:")
        add("            ctx.add_basemap(ax, source=basemap, crs='EPSG:3857', reset_extent=False)")
        add("            if isinstance(basemap, str) and ax.images:  # GeoTIFF: keep the padding inside the raster")
        add("                left, right, bottom, top = ax.images[-1].get_extent()")
        add("                xlim = (max(xlim[0], left) if left <= x.min() else xlim[0],")
        add("                        min(xlim[1], right) if right >= x.max() else xlim[1])")
        add("                ylim = (max(ylim[0], bottom) if bottom <= y.min() else ylim[0],")
        add("                        min(ylim[1], top) if top >= y.max() else ylim[1])")
        add("        except Exception as exc:  # offline, or a GeoTIFF that does not cover the points")
        add("            print('Basemap not drawn:', exc)")
        add("        ax.set_xlim(xlim)")
        add("        ax.set_ylim(ylim)")
        add("    ax.set_title(title)")
        add("    ax.set_xlabel('Longitude')")
        add("    ax.set_ylabel('Latitude')")
        add("    ax.set_xticks([])")
        add("    ax.set_yticks([])")
        add("    plt.tight_layout()")
        add("    plt.show()")
        add("")
        add("")
        add(f"BASEMAP = {basemap_expr}")
        if db:
            add(f"plot_clustered_locations(df, 'DBSCAN_Cluster', {titles[0]!r}, BASEMAP)")
        if hd:
            add(f"plot_clustered_locations(df, 'HDBSCAN_Cluster', {titles[1]!r}, BASEMAP)")
    return "\n".join(lines) + "\n"


# =============================================================================
# UI: sidebar (data source)
# =============================================================================
def sidebar_data() -> Loaded | None:
    sb = st.sidebar
    sb.header("1 · Data")
    files = list_data_files(str(DATA_DIR))
    sources = (["data/ folder"] if files else []) + ["Upload", "URL"]
    kind = sb.radio("Source", sources, horizontal=True, key="source_kind")

    if kind == "data/ folder":
        rel = sb.selectbox(
            "File",
            files,
            index=default_file_index(files),
            key="folder_file",
            help="CSV, TSV, .csv.gz or .zip files found in the data/ folder next to app.py.",
        )
        path = DATA_DIR / rel
        try:
            raw = path.read_bytes()
        except OSError as exc:
            sb.error(f"Cannot read {rel}: {exc}")
            return None
        source = {"kind": "path", "ref": f"data/{rel}", "name": path.name}
    elif kind == "Upload":
        upload = sb.file_uploader(
            "CSV / TSV file (also zipped or gzipped)", type=["csv", "tsv", "txt", "zip", "gz"], key="upload"
        )
        if upload is None:
            return None
        raw = upload.getvalue()
        source = {"kind": "upload", "ref": upload.name, "name": upload.name}
    else:
        with sb.form("url_form"):
            typed = st.text_input(
                "Direct link to a CSV, .gz or .zip file",
                value=st.session_state.get("url_value", ODCAF_ZIP_URL),
                help="Pre-filled with the ODCAF archive link given in the notebook. GitHub 'blob' links are "
                "converted to raw links.",
            )
            submitted = st.form_submit_button("Load")
        if submitted:
            st.session_state["url_value"] = typed
            st.session_state["url_loaded"] = normalise_url(typed)
        url = st.session_state.get("url_loaded")
        if not url:
            sb.caption("Press **Load** to download the file.")
            return None
        try:
            raw = fetch_url(url)
        except Exception as exc:
            sb.error(f"Download failed: {exc}")
            return None
        name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "data.csv"
        source = {"kind": "url", "ref": url, "name": name}

    if not raw:
        sb.error("The file is empty.")
        return None
    container = detect_container(raw)
    member = None
    if container == "zip":
        try:
            members = [m for m, _ in zip_members(raw)]
        except zipfile.BadZipFile as exc:
            sb.error(f"Not a valid zip archive: {exc}")
            return None
        if not members:
            sb.error("The archive contains no CSV or TSV file.")
            return None
        member = members[0]
        if len(members) > 1:
            member = sb.selectbox("File inside the archive", members, key=f"member:{source['ref']}")
    try:
        data = unpack(raw, container, member)
    except Exception as exc:
        sb.error(f"Cannot extract the file: {exc}")
        return None

    detected = detect_format(data)
    file_key = _hash(source["ref"], member)
    with sb.expander("Read options"):
        enc = st.selectbox(
            "Encoding",
            ["auto"] + ENCODINGS,
            format_func=lambda e: f"auto ({detected['encoding']})" if e == "auto" else e,
            key=f"read:{file_key}:encoding",
        )
        sep = st.selectbox(
            "Delimiter",
            ["auto"] + list(DELIMITERS),
            format_func=lambda d: f"auto ({DELIMITERS[detected['sep']]})" if d == "auto" else DELIMITERS[d],
            key=f"read:{file_key}:sep",
        )
        dec = st.selectbox(
            "Decimal separator",
            ["auto", ".", ","],
            format_func=lambda d: f"auto ('{detected['decimal']}')" if d == "auto" else f"'{d}'",
            key=f"read:{file_key}:decimal",
        )
        header = st.checkbox(
            "The first row holds the column names",
            value=detected["header"],
            key=f"read:{file_key}:header",
            help="Detected automatically: a first row made only of numbers means the file has no column names.",
        )
        markers = st.text_input(
            "Extra missing-value markers (comma-separated)",
            value=DEFAULT_MISSING_MARKERS,
            key=f"read:{file_key}:markers",
            help="ODCAF uses '..' for missing values (notebook, Implementation 1). Empty cells, NA, N/A, "
            "null, NaN and similar are always treated as missing.",
        )
    read = {
        "encoding": detected["encoding"] if enc == "auto" else enc,
        "sep": detected["sep"] if sep == "auto" else sep,
        "decimal": detected["decimal"] if dec == "auto" else dec,
        "header": bool(header),
        "na_values": [t.strip() for t in markers.split(",") if t.strip()],
    }
    if read["sep"] == read["decimal"]:
        sb.error("The delimiter and the decimal separator must be different.")
        return None
    try:
        df, skipped = parse_csv(
            data, read["encoding"], read["sep"], read["decimal"], read["header"], tuple(read["na_values"])
        )
    except Exception as exc:
        sb.error(f"Could not read the file as CSV: {exc}")
        return None
    if df.empty or df.shape[1] == 0:
        sb.error("No rows were found in the file.")
        return None
    read["skip_bad_lines"] = skipped
    if skipped:
        sb.warning("Some malformed lines were skipped while reading.")
    inner = f" › {Path(member).name}" if member else ""
    sb.success(f"**{source['name']}{inner}**  \n{len(df):,} rows × {df.shape[1]} columns")
    preset = is_notebook_dataset(df)
    if preset:
        sb.caption(
            "📓 This is the notebook's ODCAF dataset: the defaults reproduce the notebook (museums, "
            "Latitude × 2, DBSCAN eps 1.0 / min_samples 3, HDBSCAN min_cluster_size 3)."
        )
    source.update(container=container, member=member, read=read)
    key = _hash(source["ref"], member, len(raw), tuple(map(str, df.columns)))
    return Loaded(df=df, key=key, name=source["name"], preset=preset, source=source)


# =============================================================================
# UI: tabs
# =============================================================================
def welcome() -> None:
    st.info("Load a CSV file in the sidebar to start: upload one, paste a link, or put files in the app's `data/` folder.")
    st.markdown(
        """
**The notebook's pipeline, for any CSV file**

| Notebook step | In this app |
|---|---|
| `pd.read_csv(..., encoding="ISO-8859-1")` | Upload, URL or `data/` folder; encoding, delimiter and decimal detected; `.zip`/`.gz` supported |
| `df.head()`, `value_counts()`, missing values as `'..'` | **Data** tab, plus configurable missing-value markers |
| `df[df.ODCAF_Facility_Type == 'museum']` | **Prepare** › filter on any columns |
| `df[['Latitude', 'Longitude']]`, `astype('float')` | **Prepare** › any numeric (and one-hot categorical) features, lenient number parsing |
| `coords_scaled['Latitude'] = 2 * ...` | **Prepare** › scaling, per-feature weights or haversine distance |
| `DBSCAN(eps=1.0, min_samples=3)` | **DBSCAN** tab, with a k-distance plot that suggests eps |
| `hdbscan.HDBSCAN(min_cluster_size=3)` | **HDBSCAN** tab (hdbscan library or scikit-learn) |
| `plot_clustered_locations(...)` on Canada.tif | Interactive map for any region, or the same static matplotlib map |
| Closing remarks | **Compare** tab, then **Export** labels and a reproducible script |
"""
    )


def explore_tab(loaded: Loaded) -> None:
    df, key = loaded.df, loaded.key
    profile = profile_columns(df)
    numeric_cols = numeric_like_columns(profile)
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", f"{df.shape[1]:,}")
    c[2].metric("Numeric columns", f"{len(numeric_cols):,}", help="At least 90% of the non-empty values are numbers.")
    c[3].metric("Missing cells", f"{100 * float(df.isna().to_numpy().mean()):.1f}%")

    st.markdown("#### First rows  ·  `df.head()`")
    rows = st.slider("Rows to show", 5, 100, 10, key=f"{key}:head")
    st.dataframe(df.head(rows), **FILL_DF)

    st.markdown("#### Columns  ·  `df.info()`")
    st.caption(
        "**numeric %** = share of the non-empty values that read as numbers. "
        + NOTEBOOK_NOTES["missing"]
    )
    st.dataframe(
        profile,
        hide_index=True,
        column_config={
            "missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f"),
            "numeric %": st.column_config.ProgressColumn("numeric %", min_value=0, max_value=100, format="%.0f"),
        },
        **FILL_DF,
    )

    st.markdown("#### Value counts  ·  `df[column].value_counts()`")
    columns = list(df.columns)
    if loaded.preset:
        default = columns.index("ODCAF_Facility_Type")
    else:
        candidates = [i for i, col in enumerate(columns) if col not in numeric_cols and 2 <= profile.loc[str(col), "unique"] <= 50]
        default = candidates[0] if candidates else 0
    col = st.selectbox("Column", columns, index=default, key=f"{key}:vc")
    counts = df[col].value_counts(dropna=False)
    label = str(col) if str(col) != "rows" else "value"
    table = pd.DataFrame(
        {label: ["(missing)" if pd.isna(v) else str(v) for v in counts.index], "rows": counts.to_numpy()}
    )
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(table, hide_index=True, height=360, **FILL_DF)
        st.caption(f"{len(counts):,} distinct values")
    with right:
        top = table.head(25).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=top["rows"],
                y=top[label],
                orientation="h",
                marker_color=SERIES[theme_type()][0],
                hovertemplate="%{y}: %{x:,}<extra></extra>",
            )
        )
        fig.update_layout(
            height=max(260, 26 * len(top) + 60),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="rows",
            yaxis=dict(type="category"),
        )
        st.plotly_chart(fig, key=f"{key}:vc_fig", **FILL_PLOTLY)


def prepare_tab(loaded: Loaded) -> Prepared | None:
    df, dkey, preset = loaded.df, loaded.key, loaded.preset
    profile = profile_columns(df)
    numeric_cols = numeric_like_columns(profile)

    def k(name: str) -> str:
        return f"{dkey}:prep:{name}"

    # ① filter ---------------------------------------------------------------
    st.subheader("① Filter the rows")
    st.caption(
        "Generalises the notebook's `df[df.ODCAF_Facility_Type == 'museum']`. Pick columns to filter on; "
        "leave the list empty to keep every row."
    )
    default_filters = ["ODCAF_Facility_Type"] if preset else []
    filter_cols = st.multiselect("Filter on columns", list(df.columns), default=default_filters, key=k("filter_cols"))
    mask = pd.Series(True, index=df.index)
    filters = []
    for col in filter_cols:
        s = df[col]
        unique = int(profile.loc[str(col), "unique"])
        if col in numeric_cols and unique > 25:
            num = to_numeric_lenient(s)
            lo, hi = float(num.min()), float(num.max())
            if lo == hi:
                st.caption(f"`{col}` has a single value ({lo:g}); nothing to filter.")
                continue
            integer = bool(np.allclose(num.dropna(), np.round(num.dropna()))) and max(abs(lo), abs(hi)) < 1e12
            if integer:
                lo, hi = int(lo), int(hi)
                chosen = st.slider(f"Keep rows where `{col}` is between", lo, hi, (lo, hi), step=1, key=k(f"range:{col}"))
            else:
                chosen = st.slider(
                    f"Keep rows where `{col}` is between", lo, hi, (lo, hi), step=(hi - lo) / 200, key=k(f"range:{col}")
                )
            if tuple(chosen) != (lo, hi):
                mask &= num.between(chosen[0], chosen[1])
                filters.append({"column": col, "kind": "range", "min": py_value(chosen[0]), "max": py_value(chosen[1])})
        else:
            counts = s.value_counts()
            options = [py_value(v) for v in counts.index[:MAX_FILTER_OPTIONS]]
            lookup = dict(zip(options, counts.to_numpy()[:MAX_FILTER_OPTIONS]))
            default = ["museum"] if preset and col == "ODCAF_Facility_Type" and "museum" in options else []
            chosen = st.multiselect(
                f"Keep rows where `{col}` is one of",
                options,
                default=default,
                format_func=lambda v, lookup=lookup: f"{v}  ({lookup.get(v, 0):,})",
                key=k(f"values:{col}"),
                help=None if len(counts) <= MAX_FILTER_OPTIONS else f"Showing the {MAX_FILTER_OPTIONS} most frequent values.",
            )
            if chosen:
                mask &= s.isin(chosen)
                filters.append({"column": col, "kind": "values", "values": list(chosen)})
    filtered = df[mask]
    st.caption(f"**{len(filtered):,}** of {len(df):,} rows kept.")
    if filtered.empty:
        st.warning("No rows are left after filtering.")
        return None

    # ② features -------------------------------------------------------------
    st.subheader("② Select the features")
    st.caption(
        "Generalises `df[['Latitude', 'Longitude']]` and the conversion to float. Numeric columns stored as "
        "text are converted leniently (spaces and trailing ',' or ';' are ignored); rows with a missing or "
        "invalid value in any selected feature are dropped, like the notebook's `df[df.Latitude != '..']`."
    )
    if not numeric_cols:
        st.warning(
            "No numeric column was found. Check the **Read options** (delimiter, decimal separator, missing-value "
            "markers) or use categorical features below."
        )
    lat_guess, lon_guess = guess_lat_lon(df, numeric_cols)
    if preset:
        default_features = ["Latitude", "Longitude"]
    elif lat_guess and lon_guess:
        default_features = [lat_guess, lon_guess]
    else:
        default_features = [
            c for c in numeric_cols if not looks_like_id(c, df[c]) and int(profile.loc[str(c), "unique"]) > 1
        ][:5]
    features = st.multiselect(
        "Numeric features",
        numeric_cols,
        default=[c for c in default_features if c in numeric_cols],
        key=k("features"),
    )
    categorical_options = [
        c for c in df.columns if c not in numeric_cols and 2 <= int(profile.loc[str(c), "unique"]) <= 50
    ]
    with st.expander("Categorical features (optional, one-hot encoded)"):
        categorical = st.multiselect(
            "Categorical features",
            categorical_options,
            default=[],
            key=k("categorical"),
            help="Columns with 2 to 50 distinct values. Each value becomes a 0/1 column.",
        )
    if not features and not categorical:
        st.info("Select at least one feature to continue.")
        return None

    # ③ coordinates ----------------------------------------------------------
    st.subheader("③ Geographic coordinates")
    lat_col = lon_col = None
    use_geo = st.checkbox(
        "The data have latitude/longitude columns (enables maps and the haversine distance)",
        value=bool(lat_guess and lon_guess),
        key=k("geo"),
        disabled=len(numeric_cols) < 2,
    )
    if use_geo and len(numeric_cols) >= 2:
        c1, c2 = st.columns(2)
        lat_index = numeric_cols.index(lat_guess) if lat_guess in numeric_cols else 0
        lon_index = numeric_cols.index(lon_guess) if lon_guess in numeric_cols else min(1, len(numeric_cols) - 1)
        lat_pick = c1.selectbox("Latitude column", numeric_cols, index=lat_index, key=k("lat"))
        lon_pick = c2.selectbox("Longitude column", numeric_cols, index=lon_index, key=k("lon"))
        problem = coordinate_problem(df, lat_pick, lon_pick)
        if problem:
            st.warning(problem + " Maps are disabled.")
        else:
            lat_col, lon_col = lat_pick, lon_pick

    work = filtered.copy()
    for col in features + [c for c in (lat_col, lon_col) if c and c not in features]:
        work[col] = to_numeric_lenient(work[col])
    clean = work.dropna(subset=features + categorical)
    dropped = len(work) - len(clean)
    st.caption(
        f"**{len(clean):,}** rows have valid values for every selected feature "
        f"({dropped:,} dropped because of missing or non-numeric values)."
    )
    if len(clean) < 3:
        st.warning("At least 3 complete rows are needed for clustering.")
        return None

    # ④ distance and scaling -------------------------------------------------
    st.subheader("④ Distance and scaling")
    latlon_only = lat_col is not None and set(features) == {lat_col, lon_col} and not categorical
    metric_options = ["euclidean", "manhattan", "chebyshev"] + (["haversine"] if latlon_only else [])
    metric = st.selectbox(
        "Distance metric (used by DBSCAN and HDBSCAN)",
        metric_options,
        format_func=METRICS.get,
        key=k(f"metric:{latlon_only}"),
        help="Haversine is offered when the features are exactly the latitude and longitude columns.",
    )
    feature_weights = {}
    if metric == "haversine":
        scaling = "none"
        st.info(
            "Haversine measures real distances on the Earth's surface. The coordinates are converted to "
            "radians and every distance parameter (eps, cluster_selection_epsilon) is entered in **kilometres**. "
            "No scaling is needed."
        )
    else:
        default_scaling = "none" if latlon_only else "standard"
        scaling = st.selectbox(
            "Scaling",
            list(SCALINGS),
            index=list(SCALINGS).index(default_scaling),
            format_func=SCALINGS.get,
            key=k(f"scaling:{latlon_only}"),
            help="DBSCAN and HDBSCAN are distance based, so features on larger scales dominate unless scaled.",
        )
        if latlon_only:
            st.caption(NOTEBOOK_NOTES["scaling_geo"])
        names = features + categorical
        with st.expander("Feature weights (multipliers applied after scaling)", expanded=latlon_only):
            st.caption("A weight of 2 counts that feature's differences twice; 0 ignores it.")
            columns = st.columns(min(4, len(names)))
            for i, name in enumerate(names):
                default_weight = 2.0 if (latlon_only and name == lat_col) else 1.0
                feature_weights[name] = float(
                    columns[i % len(columns)].number_input(
                        f"{name}", min_value=0.0, value=default_weight, step=0.5, key=k(f"weight:{latlon_only}:{name}")
                    )
                )

    sample_n = None
    if len(clean) > LARGE_DATA_ROWS:
        with st.expander("Large dataset", expanded=True):
            use_sample = stateful(
                st.checkbox,
                f"Cluster a random sample instead of all {len(clean):,} rows (faster)",
                k("sample_on"),
                len(clean) > 50_000,
            )
            size = stateful(st.number_input, "Sample size", k("sample_n"), DEFAULT_SAMPLE_ROWS, min_value=1000, step=1000)
        if use_sample and int(size) < len(clean):
            sample_n = int(size)
            clean = clean.sample(n=sample_n, random_state=0).sort_index()

    model_weights = {}
    for name, weight in feature_weights.items():
        if weight == 1.0:
            continue
        if name in categorical:
            for dummy in pd.get_dummies(clean[[name]], columns=[name], dtype=float).columns:
                model_weights[str(dummy)] = weight
        else:
            model_weights[name] = weight
    X, model_columns = build_matrix(clean, features, categorical, metric, scaling, model_weights, lat_col, lon_col)
    if X.shape[1] == 0 or not np.isfinite(X).all():
        st.error("The model input is empty or contains invalid values.")
        return None

    st.subheader("Model input")
    st.caption(
        f"The notebook's `coords_scaled`: **{X.shape[0]:,} rows × {X.shape[1]} columns** passed to DBSCAN and "
        f"HDBSCAN{' (radians)' if metric == 'haversine' else ''}."
    )
    st.dataframe(pd.DataFrame(X[:8], columns=model_columns, index=clean.index[:8]), **FILL_DF)
    if features:
        with st.expander("Summary statistics of the selected features  ·  `df.describe()`"):
            st.dataframe(clean[features].describe().T, **FILL_DF)

    space_key = _hash(dkey, features, categorical, metric, scaling)
    recipe = {
        "source": loaded.source,
        "filters": filters,
        "features": list(features),
        "categorical": list(categorical),
        "lat_col": lat_col,
        "lon_col": lon_col,
        "sample_n": sample_n,
        "metric": metric,
        "scaling": scaling,
        "weights": model_weights,
        "is_notebook": preset,
    }
    return Prepared(
        data=clean,
        X=X,
        model_columns=model_columns,
        features=list(features),
        categorical=list(categorical),
        metric=metric,
        scaling=scaling,
        lat_col=lat_col,
        lon_col=lon_col,
        space_key=space_key,
        is_notebook=preset,
        recipe=recipe,
    )


def display_settings(prep: Prepared, dkey: str) -> dict:
    """Plot options, rendered inside a sidebar container (call within `with container:`)."""
    sb = st
    sb.header("2 · Display")

    def k(name: str) -> str:
        return f"{dkey}:view:{name}"

    view = {"theme": theme_type()}
    kinds = ["map"] if prep.has_geo else []
    if len(prep.features) >= 2:
        kinds.append("pair")
    if prep.X.shape[1] >= 2:
        kinds.append("pca2")
    if prep.X.shape[1] >= 3:
        kinds.append("pca3")
    if prep.X.shape[1] == 1:
        kinds.append("strip")
    if not prep.has_geo and "pair" in kinds and prep.X.shape[1] > 2:
        kinds.remove("pca2")
        kinds.insert(0, "pca2")
    view["kind"] = sb.selectbox("Plot", kinds, format_func=VIEW_LABELS.get, key=k("kind:" + ",".join(kinds)))
    if view["kind"] == "map":
        view["map_style"] = sb.selectbox(
            "Interactive basemap",
            list(MAP_STYLES),
            format_func=MAP_STYLES.get,
            key=k("map_style"),
            help="Map tiles are downloaded from the internet. If the map stays blank (offline), choose 'No basemap'.",
        )
    if view["kind"] == "pair":
        view["x"] = sb.selectbox("X axis", prep.features, index=0, key=k("x:" + ",".join(prep.features)))
        view["y"] = sb.selectbox("Y axis", prep.features, index=1, key=k("y:" + ",".join(prep.features)))
    view["coloring"] = sb.radio(
        "Cluster colours",
        ["each", "top"],
        format_func={"each": "Every cluster (8 colours, repeated)", "top": "Largest 8 clusters + 'Other'"}.get,
        key=k("coloring"),
        help="Colours repeat when there are more than 8 clusters: hover a point to read its cluster.",
    )
    n = len(prep.X)
    view["point_size"] = stateful(
        sb.slider, "Point size", k("size"), 8 if n <= 3000 else 6 if n <= 20000 else 4, min_value=2, max_value=16
    )
    hover_options = [c for c in prep.data.columns if c not in RESERVED_PLOT_COLUMNS]
    view["hover"] = sb.multiselect(
        "Details shown on hover",
        hover_options,
        default=guess_hover_columns(hover_options),
        key=k("hover"),
    )
    view["static_basemap"] = "none"
    if prep.has_geo and plt is not None:
        with st.expander("Notebook-style static map"):
            tifs = list_geotiffs(str(DATA_DIR))
            options = [f"file:{DATA_DIR / t}" for t in tifs] + ["upload"] + [f"online:{name}" for name in ONLINE_BASEMAPS] + ["none"]
            labels = {f"file:{DATA_DIR / t}": f"GeoTIFF: data/{t}" for t in tifs}
            labels.update({"upload": "Upload a GeoTIFF…", "none": "No basemap"})
            labels.update({f"online:{name}": name for name in ONLINE_BASEMAPS})
            if prep.is_notebook and tifs:
                default = 0
            else:
                default = options.index("online:Online: CARTO Positron")
            choice = st.selectbox(
                "Basemap",
                options,
                index=default,
                format_func=labels.get,
                key=k("static_basemap"),
                disabled=ctx is None,
                help="The notebook uses data/Canada.tif. A GeoTIFF only helps if it covers your points; online "
                "tiles need an internet connection.",
            )
            if ctx is None:
                st.caption("Install `contextily` to draw basemaps; points are drawn without one.")
                choice = "none"
            if choice == "upload":
                upload = st.file_uploader("GeoTIFF basemap", type=["tif", "tiff"], key=k("tif_upload"))
                choice = "none"
                if upload is not None:
                    data = upload.getvalue()
                    path = Path(tempfile.gettempdir()) / f"basemap_{hashlib.sha1(data).hexdigest()[:12]}.tif"
                    if not path.exists():
                        path.write_bytes(data)
                    choice = f"file:{path}"
            view["static_basemap"] = choice
    return view


def results_block(prep: Prepared, labels: np.ndarray, algo: str, view: dict, k, core=None, probabilities=None) -> None:
    summary = summarize(labels)
    quality = cluster_quality(prep.X, labels, prep.metric)
    m = st.columns(4)
    m[0].metric("Clusters", f"{summary['clusters']:,}")
    m[1].metric("Noise points (-1)", f"{summary['noise']:,} ({summary['noise_pct']:.1f}%)")
    m[2].metric("Largest cluster", f"{summary['largest']:,} ({summary['largest_pct']:.1f}%)")
    m[3].metric(
        "Silhouette",
        fmt_num(quality["silhouette"]),
        help="Mean silhouette of the clustered points (noise excluded), from -1 to 1; higher means tighter, "
        "better separated clusters.",
    )
    notes = []
    if core is not None:
        border = int(((labels != -1) & ~core).sum())
        notes.append(f"{int(core.sum()):,} core points · {border:,} border points · {summary['noise']:,} noise points")
    if probabilities is not None and summary["clusters"]:
        notes.append(f"mean membership probability of clustered points: {probabilities[labels != -1].mean():.2f}")
    if notes:
        st.caption("  ·  ".join(notes))

    title = f"{algo}: {summary['clusters']} clusters, {summary['noise']} noise points"
    fig, n_groups = cluster_figure(prep, labels, view, title, probabilities)
    st.plotly_chart(fig, key=k("fig"), **FILL_PLOTLY)
    if n_groups > MAX_LEGEND_ITEMS:
        st.caption(
            f"Legend hidden for {n_groups} groups: hover a point to read its cluster, or choose "
            "*Largest 8 clusters + 'Other'* in the sidebar."
        )

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Cluster sizes**  ·  `df['Cluster'].value_counts()`")
        st.dataframe(size_table(labels), hide_index=True, height=360, **FILL_DF)
    with right:
        st.plotly_chart(size_bar_figure(labels, view["theme"]), key=k("sizes"), **FILL_PLOTLY)

    with st.expander("Cluster profiles (mean feature values per cluster)"):
        st.dataframe(profile_table(prep, labels), hide_index=True, **FILL_DF)
    with st.expander("Quality metrics"):
        st.dataframe(
            pd.DataFrame(
                {
                    "metric": ["Silhouette (higher is better)", "Davies–Bouldin (lower is better)", "Calinski–Harabasz (higher is better)"],
                    "value": [fmt_num(quality["silhouette"]), fmt_num(quality["davies_bouldin"]), fmt_num(quality["calinski_harabasz"])],
                }
            ),
            hide_index=True,
            **FILL_DF,
        )
        st.caption(
            "Computed on clustered points only (noise excluded). These indices favour compact, round clusters, "
            "so treat them as a rough guide for density-based models."
        )
    if probabilities is not None and summary["clusters"]:
        with st.expander("Membership probabilities"):
            st.plotly_chart(probability_figure(probabilities, labels, view["theme"]), key=k("proba"), **FILL_PLOTLY)
            st.caption("HDBSCAN's confidence that each point belongs to its cluster (1 = core of the cluster).")
    if prep.has_geo and plt is not None:
        with st.expander("Notebook-style static map  ·  `plot_clustered_locations(df, title)`"):
            if st.checkbox("Draw the static map", value=prep.is_notebook, key=k("static_on")):
                if prep.is_notebook:
                    title = "Museums Clustered by Proximity" if algo == "DBSCAN" else "Museums Hierarchically Clustered by Proximity"
                else:
                    title = f"{algo} clusters"
                png, warning = static_map_png(
                    prep.data[prep.lat_col].to_numpy(dtype=float),
                    prep.data[prep.lon_col].to_numpy(dtype=float),
                    np.asarray(labels),
                    title,
                    view["static_basemap"],
                    float(round((view["point_size"] * 0.7) ** 2, 1)),
                )
                if warning:
                    st.warning(warning)
                st.image(png, **FILL_IMAGE)


def dbscan_tab(prep: Prepared, view: dict) -> dict | None:
    with st.expander("How DBSCAN works (notebook notes)"):
        st.markdown(NOTEBOOK_NOTES["dbscan"])

    def k(name: str) -> str:
        return f"{prep.space_key}:dbscan:{name}"

    n, d = prep.X.shape
    c1, c2, c3 = st.columns([1, 1, 1.2])
    default_min_samples = 3 if prep.is_notebook else int(min(max(4, 2 * d), 20))
    min_samples = int(
        stateful(
            c1.number_input, "min_samples", k("min_samples"), default_min_samples, min_value=1, step=1,
            help="Minimum number of samples needed to form a neighbourhood (the point itself included).",
        )
    )
    kd = k_distances(prep.X, max(2, min_samples), prep.metric)
    suggestion = nice_number(suggest_eps(kd))
    eps_key = k("eps")
    if eps_key not in st.session_state:
        notebook_default = prep.is_notebook and prep.metric == "euclidean" and prep.scaling == "none"
        initial = 1.0 if notebook_default else suggestion
        st.session_state[eps_key] = float(initial)
        st.session_state[k("eps_scale")] = float(initial)
    scale = st.session_state.get(k("eps_scale"), suggestion)
    eps = float(
        c2.number_input(
            f"eps ({prep.unit})", min_value=0.0, step=step_for(scale), format=format_for(scale), key=eps_key,
            help="Neighbourhood search radius.",
        )
    )
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c3.button(
        f"Use suggested eps ≈ {fmt_num(suggestion)}",
        on_click=_set_state,
        args=(eps_key, float(suggestion)),
        key=k("use_suggestion"),
        help="Knee of the k-distance curve below.",
    )
    core_share = float((kd <= eps).mean()) if eps > 0 else 0.0
    with st.expander("Choosing eps: the k-distance plot", expanded=not prep.is_notebook):
        st.plotly_chart(
            kdistance_figure(kd, eps, suggestion, max(2, min_samples), prep.unit, view["theme"]),
            key=k("kdist"),
            **FILL_PLOTLY,
        )
        st.caption(
            f"Each point's distance to its {max(2, min_samples)}-th nearest neighbour (itself included), sorted. "
            f"Points under the eps line are core points: **{100 * core_share:.0f}%** at the current eps. "
            "A good eps sits near the knee, where the curve starts to rise steeply. The y-axis stops at the "
            "99.5th percentile."
        )
    if eps <= 0:
        st.error("eps must be greater than 0.")
        return None
    if min_samples > n:
        st.error(f"min_samples ({min_samples}) is larger than the number of rows ({n}).")
        return None
    eps_model = prep.to_model_distance(eps)
    pairs = estimate_neighbor_pairs(prep.X, eps_model, prep.metric)
    if pairs > MAX_NEIGHBOR_PAIRS:
        st.error(
            f"With eps = {fmt_num(eps)} each point has about {pairs / n:,.0f} neighbours ({pairs:,.0f} pairs in "
            "total), which needs too much memory. Lower eps or cluster a random sample (Prepare tab)."
        )
        return None
    with st.spinner("Running DBSCAN…"):
        labels, core = run_dbscan(prep.X, eps_model, min_samples, prep.metric)
    st.divider()
    results_block(prep, labels, "DBSCAN", view, k, core=core)
    return {"labels": labels, "params": {"eps": eps, "min_samples": min_samples}}


def hdbscan_tab(prep: Prepared, view: dict) -> dict | None:
    backends = [b for b, ok in (("hdbscan", hdbscan_lib is not None), ("sklearn", SkHDBSCAN is not None)) if ok]
    if not backends:
        st.error("HDBSCAN needs the `hdbscan` package or scikit-learn 1.3 or later.")
        return None
    with st.expander("How HDBSCAN works (notebook notes)"):
        st.markdown(NOTEBOOK_NOTES["hdbscan"])

    def k(name: str) -> str:
        return f"{prep.space_key}:hdbscan:{name}"

    n = len(prep.X)
    c1, c2, c3 = st.columns(3)
    min_cluster_size = int(
        stateful(
            c1.number_input, "min_cluster_size", k("min_cluster_size"), 3 if prep.is_notebook else 5, min_value=2, step=1,
            help="Smallest group of points that counts as a cluster.",
        )
    )
    auto = c2.checkbox("min_samples = None (same as min_cluster_size)", value=True, key=k("min_samples_none"))
    min_samples = None
    if not auto:
        min_samples = int(
            stateful(
                c2.number_input, "min_samples", k("min_samples"), min_cluster_size, min_value=1, step=1,
                help="Larger values make the clustering more conservative: more points become noise.",
            )
        )
    backend = c3.selectbox("Implementation", backends, format_func=BACKENDS.get, key=k("backend"))
    with st.expander("More parameters"):
        a1, a2, a3 = st.columns(3)
        scale = st.session_state.get(f"{prep.space_key}:dbscan:eps_scale", 1.0)
        selection_eps = float(
            a1.number_input(
                f"cluster_selection_epsilon ({prep.unit})", min_value=0.0, value=0.0, step=step_for(scale),
                format=format_for(scale), key=k("selection_eps"),
                help="Clusters closer than this distance are merged (0 = off).",
            )
        )
        method = a2.selectbox(
            "cluster_selection_method", ["eom", "leaf"], key=k("method"),
            help="'eom' keeps the most stable clusters; 'leaf' returns the smallest, most homogeneous ones.",
        )
        single = a3.checkbox("allow_single_cluster", value=False, key=k("single"))
    if backend == "sklearn":
        st.caption(
            "scikit-learn counts the point itself in `min_samples`, the hdbscan library does not: the app passes "
            "`min_samples + 1` so both implementations follow the notebook's convention."
        )
    effective = (min_samples if min_samples is not None else min_cluster_size) + (1 if backend == "sklearn" else 0)
    if min_cluster_size > n or effective >= n:
        st.error(f"min_cluster_size / min_samples are too large for {n} rows.")
        return None
    try:
        with st.spinner("Running HDBSCAN…"):
            labels, probabilities = run_hdbscan(
                prep.X, backend, min_cluster_size, min_samples, prep.metric,
                prep.to_model_distance(selection_eps), method, single,
            )
    except Exception as exc:
        st.error(f"HDBSCAN failed: {exc}")
        return None
    st.divider()
    results_block(prep, labels, "HDBSCAN", view, k, probabilities=probabilities)
    params = {
        "backend": backend,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_epsilon": selection_eps,
        "cluster_selection_method": method,
        "allow_single_cluster": bool(single),
    }
    return {"labels": labels, "probabilities": probabilities, "params": params}


def compare_tab(prep: Prepared, db: dict | None, hdb: dict | None, view: dict) -> None:
    if db is None or hdb is None:
        st.info("Run both models (DBSCAN and HDBSCAN tabs) to compare them.")
        return
    a, b = db["labels"], hdb["labels"]
    st.markdown(NOTEBOOK_NOTES["compare"])
    sa, sb_ = summarize(a), summarize(b)
    qa, qb = cluster_quality(prep.X, a, prep.metric), cluster_quality(prep.X, b, prep.metric)
    stats = agreement(a, b)

    c = st.columns(3)
    c[0].metric("Adjusted Rand index", f"{stats['ari']:.3f}", help="Agreement between the two labelings: 1 = identical, about 0 = chance. Noise counts as one group.")
    c[1].metric("Adjusted mutual information", fmt_num(stats["ami"]), help="Another agreement score (1 = identical). Skipped above 30,000 points.")
    c[2].metric("Noise in both models", f"{int(((a == -1) & (b == -1)).sum()):,}")

    rows = [
        ("Clusters", sa["clusters"], sb_["clusters"]),
        ("Noise points", sa["noise"], sb_["noise"]),
        ("Noise %", round(sa["noise_pct"], 1), round(sb_["noise_pct"], 1)),
        ("Largest cluster (points)", sa["largest"], sb_["largest"]),
        ("Largest cluster (% of points)", round(sa["largest_pct"], 1), round(sb_["largest_pct"], 1)),
        ("Median cluster size", sa["median"], sb_["median"]),
        ("Smallest cluster", sa["smallest"], sb_["smallest"]),
        ("Silhouette", fmt_num(qa["silhouette"]), fmt_num(qb["silhouette"])),
        ("Davies–Bouldin", fmt_num(qa["davies_bouldin"]), fmt_num(qb["davies_bouldin"])),
        ("Calinski–Harabasz", fmt_num(qa["calinski_harabasz"]), fmt_num(qb["calinski_harabasz"])),
    ]
    table = pd.DataFrame(rows, columns=["Metric", "DBSCAN", "HDBSCAN"]).astype(str)
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Side by side**")
        st.dataframe(table, hide_index=True, **FILL_DF)
        st.markdown("**Who is noise?** (number of points)")
        st.dataframe(noise_table(a, b), **FILL_DF)
    with right:
        st.markdown("**Cluster size distribution**")
        st.plotly_chart(size_distribution_figure({"DBSCAN": a, "HDBSCAN": b}, view["theme"]), key=f"{prep.space_key}:cmp:dist", **FILL_PLOTLY)

    st.markdown("**How HDBSCAN splits DBSCAN's largest clusters**")
    st.dataframe(split_table(a, b), hide_index=True, **FILL_DF)

    findings = []
    if sa["clusters"] and sb_["clusters"]:
        findings.append(
            f"DBSCAN's largest cluster holds **{sa['largest_pct']:.0f}%** of the points, HDBSCAN's "
            f"**{sb_['largest_pct']:.0f}%**; median cluster sizes are {sa['median']:g} and {sb_['median']:g} points."
        )
    findings.append(
        f"HDBSCAN finds **{sb_['clusters']}** clusters against **{sa['clusters']}** for DBSCAN and labels "
        f"**{sb_['noise_pct']:.1f}%** of the points as noise against **{sa['noise_pct']:.1f}%**."
    )
    only_h = int(((a != -1) & (b == -1)).sum())
    if only_h:
        findings.append(
            f"{only_h:,} points that DBSCAN puts in a cluster are noise for HDBSCAN, typically points in sparser "
            "areas that DBSCAN's single eps still attaches to a dense group."
        )
    st.markdown("\n".join(f"- {f}" for f in findings))

    st.markdown("**The two results side by side**")
    p1, p2 = st.columns(2)
    for column, labels, name, extra in ((p1, a, "DBSCAN", None), (p2, b, "HDBSCAN", hdb["probabilities"])):
        with column:
            s = summarize(labels)
            fig, _ = cluster_figure(
                prep, labels, view, f"{name}: {s['clusters']} clusters, {s['noise']} noise", extra, height=480, width=480
            )
            st.plotly_chart(fig, key=f"{prep.space_key}:cmp:{name}", **FILL_PLOTLY)


def export_tab(prep: Prepared, db: dict | None, hdb: dict | None, loaded: Loaded, view: dict) -> None:
    out = prep.data.copy()
    if db is not None:
        out["DBSCAN_Cluster"] = db["labels"]
    if hdb is not None:
        out["HDBSCAN_Cluster"] = hdb["labels"]
        out["HDBSCAN_Probability"] = np.round(hdb["probabilities"], 4)
    stem = Path(loaded.name).stem or "data"
    st.markdown("#### Clustered data")
    st.caption(
        f"{len(out):,} clustered rows with every original column, the cluster labels (-1 = noise) and HDBSCAN's "
        "membership probability. `source_row` is the row's position in the loaded file."
    )
    st.download_button(
        "Download the clustered data (CSV)",
        out.to_csv(index_label="source_row").encode("utf-8"),
        file_name=f"{stem}_clustered.csv",
        mime="text/csv",
        key=f"{prep.space_key}:dl_csv",
    )
    st.dataframe(out.head(20), **FILL_DF)

    st.markdown("#### Reproducible Python script")
    st.caption(
        "The same pipeline as a plain script in the notebook's style (load, filter, select, clean, scale, DBSCAN, "
        f"HDBSCAN{', basemap plot' if prep.has_geo else ''}) with the parameters chosen in the app."
    )
    recipe = dict(prep.recipe)
    recipe["dbscan"] = db["params"] if db else None
    recipe["hdbscan"] = hdb["params"] if hdb else None
    recipe["static_basemap"] = view.get("static_basemap", "none")
    if recipe["static_basemap"].startswith("file:"):
        path = Path(recipe["static_basemap"][5:])
        try:
            recipe["static_basemap"] = "file:" + path.relative_to(APP_DIR).as_posix()
        except ValueError:
            recipe["static_basemap"] = "file:" + path.name
    script = generate_script(recipe)
    st.download_button(
        "Download the Python script",
        script.encode("utf-8"),
        file_name=f"{stem}_clustering.py",
        mime="text/x-python",
        key=f"{prep.space_key}:dl_py",
    )
    st.code(script, language="python")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="DBSCAN & HDBSCAN explorer", page_icon="🗺️", layout="wide")
    st.title("DBSCAN & HDBSCAN clustering explorer")
    st.caption(
        "A general version of the notebook *Comparing DBSCAN and HDBSCAN clustering*: the same pipeline for any "
        "CSV file. Data → Prepare → DBSCAN → HDBSCAN → Compare → Export."
    )
    loaded = sidebar_data()
    display_area = st.sidebar.container()
    st.sidebar.divider()
    st.sidebar.caption(
        f"HDBSCAN: {'hdbscan library ✓' if hdbscan_lib is not None else 'hdbscan library not installed'}"
        f"{' · scikit-learn ✓' if SkHDBSCAN is not None else ''}  \n"
        f"Static basemaps: {'contextily ✓' if ctx is not None else 'contextily not installed'}"
    )
    if loaded is None:
        welcome()
        return
    tabs = st.tabs(["Data", "Prepare", "DBSCAN", "HDBSCAN", "Compare", "Export"])
    with tabs[0]:
        explore_tab(loaded)
    with tabs[1]:
        prep = prepare_tab(loaded)
    if prep is None:
        for tab in tabs[2:]:
            with tab:
                st.info("Complete the **Prepare** tab first (select at least one feature with valid values).")
        return
    with display_area:
        view = display_settings(prep, loaded.key)
    with tabs[2]:
        db = dbscan_tab(prep, view)
    with tabs[3]:
        hdb = hdbscan_tab(prep, view)
    with tabs[4]:
        compare_tab(prep, db, hdb, view)
    with tabs[5]:
        export_tab(prep, db, hdb, loaded, view)


if __name__ == "__main__":
    main()
