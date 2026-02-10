import re
import time
from pathlib import Path

import numpy as np
import streamlit as st

import astropy.units as u
from astropy.table import vstack
from astroquery.nist import Nist

from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    BoxZoomTool,
    PanTool,
    WheelZoomTool,
    ResetTool,
    SaveTool,
    Range1d,
    LinearAxis,
)
from bokeh.plotting import figure
from streamlit_bokeh import streamlit_bokeh  # pip install streamlit-bokeh


# ----------------------------
# Fixed atomic species set
# ----------------------------
SPECIES = [
    # Light elements / major diagnostics
    "H I", "He I", "He II",
    "C I", "N I", "O I",
    "Na I",
    "Mg I", "Mg II",
    "Al I",
    "Si I",
    "P I",
    "S I",
    "K I",
    "Ca I", "Ca II",

    # Iron-peak / photospheric forest
    "Sc I", "Sc II",
    "Ti I", "Ti II",
    "V I", "V II",
    "Cr I", "Cr II",
    "Mn I", "Mn II",
    "Fe I", "Fe II",
    "Co I", "Co II",
    "Ni I",
    "Cu I",
    "Zn I",

    # Heavy / useful ions in optical-NIR
    "Sr II",
    "Y II",
    "Zr I", "Zr II",
    "Ba II",
    "Ce II",
    "Nd II",
]

# ----------------------------
# Robust parsing helpers
# ----------------------------
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float_scalar(v) -> float:
    """Extract the first float-like token from a value; return NaN if none."""
    if v is None:
        return np.nan
    s = str(v).strip()
    if s in ("", "--", "masked", "None"):
        return np.nan
    m = _FLOAT_RE.search(s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan


def _to_float_array(col) -> np.ndarray:
    """Vectorized-ish robust float conversion."""
    out = np.full(len(col), np.nan, dtype=float)
    for i, v in enumerate(col):
        out[i] = _to_float_scalar(v)
    return out


def _extract_wavelength_A(tab) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a wavelength array in Angstrom and a flag array indicating source column.
    Prefer Observed if numeric; otherwise use Ritz.
    flag: 1 = Observed used, 2 = Ritz used, 0 = neither
    """
    n = len(tab)
    w_obs = np.full(n, np.nan, dtype=float)
    w_ritz = np.full(n, np.nan, dtype=float)

    if "Observed" in tab.colnames:
        w_obs = _to_float_array(tab["Observed"])
    if "Ritz" in tab.colnames:
        w_ritz = _to_float_array(tab["Ritz"])

    use_obs = np.isfinite(w_obs)
    w = np.where(use_obs, w_obs, w_ritz)

    flag = np.zeros(n, dtype=np.int8)
    flag[use_obs] = 1
    flag[(~use_obs) & np.isfinite(w_ritz)] = 2
    return w, flag


# ----------------------------
# NIST query (cache + vstack-safe)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def nist_query_window_airish(wmin_A: float, wmax_A: float, species: tuple[str, ...]):
    """
    Query NIST ASD line list for a wavelength window.

    Notes:
    - astroquery.nist supports wavelength_type='vacuum' or 'vac+air' (not 'air').  (docs)
    - We request in Angstrom to keep output in Angstrom.
    - We avoid astropy vstack dtype clashes by keeping a small whitelist and coercing to str.
    """
    tabs = []
    errors = []

    wmin = float(min(wmin_A, wmax_A)) * u.AA
    wmax = float(max(wmin_A, wmax_A)) * u.AA

    KEEP = [
        "Observed", "Ritz",       # wavelength candidates
        "Rel.", "Aki", "log_gf",
        "Ei", "Ek",
        "Acc.",
        "Transition", "Type", "TP", "Line",  # occasionally helpful if present
    ]

    for sp in species:
        try:
            t = Nist.query(
                wmin,
                wmax,
                linename=sp,
                wavelength_type="vac+air",
                output_order="wavelength",
                energy_level_unit="eV",
            )
        except Exception as e:
            errors.append((sp, repr(e)))
            continue

        if t is None or len(t) == 0:
            continue

        t = t.copy()
        t["species"] = sp

        keep_cols = ["species"] + [c for c in KEEP if c in t.colnames]
        t = t[keep_cols]

        # Coerce to string so vstack won't choke on mixed dtypes across species
        for c in t.colnames:
            if c == "species":
                continue
            try:
                t[c] = t[c].astype(str)
            except Exception:
                t[c] = np.array(t[c]).astype(str)

        tabs.append(t)

    if not tabs:
        return None, errors

    return vstack(tabs, metadata_conflicts="silent"), errors


def tighten_and_rank_nist_lines(
    tab,
    wmin_A: float,
    wmax_A: float,
    *,
    T: float = 5770.0,
    max_Ei_eV: float = 7.5,
    prefer_neutral: bool = True,
):
    """
    Tighten and rank NIST lines for a "likely strong photospheric" heuristic.

    Key robustness vs earlier versions:
    - wavelength is derived row-by-row: Observed (if numeric) else Ritz
    - parsing accepts strings like '10830.34(2)' etc
    - no requirement that log_gf exists; fallback: log_gf -> log10(Aki) -> log10(Rel.+1) -> 0
    - Ei filter only applied when Ei is numeric; missing Ei does not drop the line
    - optionally exempt species from Ei penalty (He I can have high Ei but still strong)
    """
    if tab is None or len(tab) == 0:
        return None

    tab = tab.copy()

    # Derived wavelength and filter to window
    wA, wflag = _extract_wavelength_A(tab)
    good_w = np.isfinite(wA)
    tab = tab[good_w]
    wA = wA[good_w]
    wflag = wflag[good_w]

    lo = float(min(wmin_A, wmax_A))
    hi = float(max(wmin_A, wmax_A))
    inwin = (wA >= lo) & (wA <= hi)
    tab = tab[inwin]
    wA = wA[inwin]
    wflag = wflag[inwin]

    if len(tab) == 0:
        return None

    # Strength proxy
    base = np.zeros(len(tab), dtype=float)

    if "log_gf" in tab.colnames:
        lgf = _to_float_array(tab["log_gf"])
        base = np.where(np.isfinite(lgf), lgf, base)
    elif "Aki" in tab.colnames:
        aki = _to_float_array(tab["Aki"])
        with np.errstate(divide="ignore", invalid="ignore"):
            base = np.log10(aki)
        base = np.where(np.isfinite(base), base, 0.0)
    elif "Rel." in tab.colnames:
        rel = _to_float_array(tab["Rel."])
        with np.errstate(divide="ignore", invalid="ignore"):
            base = np.log10(rel + 1.0)
        base = np.where(np.isfinite(base), base, 0.0)

    # Ei penalty
    if "Ei" in tab.colnames:
        Ei_raw = _to_float_array(tab["Ei"])
    else:
        Ei_raw = np.full(len(tab), np.nan, dtype=float)

    sp = np.array([str(s) for s in tab["species"]], dtype=str)

    # species exemptions (common IR diagnostics)
    EXEMPT_HIGH_EI = {"He I", "H I"}

    keep = (~np.isfinite(Ei_raw)) | (Ei_raw <= float(max_Ei_eV)) | np.isin(sp, list(EXEMPT_HIGH_EI))
    tab = tab[keep]
    wA = wA[keep]
    wflag = wflag[keep]
    base = base[keep]
    sp = sp[keep]
    Ei_raw = Ei_raw[keep]

    if len(tab) == 0:
        return None

    Ei = np.where(np.isfinite(Ei_raw), Ei_raw, 0.0)

    kT_eV = 8.617333262e-5 * float(T)
    alpha = 0.12
    exempt = np.isin(sp, list(EXEMPT_HIGH_EI))
    alpha_eff = np.where(exempt, 0.0, alpha)

    score = base - alpha_eff * (Ei / kT_eV)

    if prefer_neutral:
        neutral_bonus = np.zeros(len(tab), dtype=float)
        neutral_bonus[np.char.find(sp.astype(str), " I") >= 0] += 0.25
        score += neutral_bonus

    # Store derived wavelength + provenance so you can display it reliably
    tab = tab.copy()
    tab["wA"] = wA
    tab["wsrc"] = np.where(wflag == 1, "Observed", "Ritz")
    tab["score"] = score

    tab.sort("score")
    tab.reverse()
    return tab


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(layout="wide", page_title="Solar Spectrum Plotter")

DATA_DIR = Path(".")
WV_FILE = DATA_DIR / "combined_fts_v5June2022_wavelength_angstrom.npy"
OBS_FILE = DATA_DIR / "combined_fts_v5June2022_observed_spectrum.npy"
ATM_FILE = DATA_DIR / "combined_fts_v5June2022_atm_absorption.npy"


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_atlas(wv_path: str, obs_path: str, atm_path: str):
    wv = np.load(wv_path).astype(np.float64)
    obs = np.load(obs_path).astype(np.float64)
    atm = np.load(atm_path).astype(np.float64)

    if not (wv.ndim == obs.ndim == atm.ndim == 1):
        raise ValueError(f"Expected 1D arrays; got shapes wv={wv.shape}, obs={obs.shape}, atm={atm.shape}")
    if not (len(wv) == len(obs) == len(atm)):
        raise ValueError(f"Length mismatch: wv={len(wv)}, obs={len(obs)}, atm={len(atm)}")

    if wv[0] > wv[-1]:
        wv = wv[::-1]
        obs = obs[::-1]
        atm = atm[::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        cor = obs / atm
    cor = np.clip(cor, 0.0, 1.05)

    return wv, obs, atm, cor


# ----------------------------
# Theme helpers
# ----------------------------
def resolve_theme_base(user_choice: str) -> str:
    c = user_choice.lower()
    if c in ("light", "dark"):
        return c
    base = st.get_option("theme.base")
    if isinstance(base, str) and base.lower() in ("light", "dark"):
        return base.lower()
    return "light"


def apply_axes_and_grid_style(p, base: str) -> dict:
    if base == "dark":
        page_bg = "#0b1220"
        panel_bg = "#111a2e"
        fg = "#e5e7eb"
        grid = "#94a3b8"
        outline = "#334155"

        p.background_fill_color = panel_bg
        p.border_fill_color = page_bg
        p.outline_line_color = outline
        p.outline_line_alpha = 0.95

        p.title.text_color = fg
        p.title.text_font_size = "14pt"

        for ax in (p.xaxis, p.yaxis):
            ax.axis_line_color = fg
            ax.major_tick_line_color = fg
            ax.minor_tick_line_color = fg
            ax.major_label_text_color = fg
            ax.axis_label_text_color = fg
            ax.major_label_text_font_size = "11pt"
            ax.axis_label_text_font_size = "12pt"

        p.xgrid.grid_line_color = grid
        p.ygrid.grid_line_color = grid
        p.xgrid.grid_line_alpha = 0.18
        p.ygrid.grid_line_alpha = 0.18

        colors = dict(
            observed="#009E73",
            telluric="#E69F00",
            corrected="#F8FAFC",
            halo="#CBD5E1",
            marker="#94a3b8",
        )
    else:
        fg = "#111827"

        p.background_fill_color = "white"
        p.border_fill_color = "white"
        p.outline_line_color = "#d1d5db"
        p.outline_line_alpha = 0.9

        p.title.text_color = fg
        p.title.text_font_size = "14pt"

        for ax in (p.xaxis, p.yaxis):
            ax.axis_line_color = fg
            ax.major_tick_line_color = fg
            ax.minor_tick_line_color = fg
            ax.major_label_text_color = fg
            ax.axis_label_text_color = fg
            ax.major_label_text_font_size = "11pt"
            ax.axis_label_text_font_size = "12pt"

        p.xgrid.grid_line_color = "#e5e7eb"
        p.ygrid.grid_line_color = "#e5e7eb"
        p.xgrid.grid_line_alpha = 0.8
        p.ygrid.grid_line_alpha = 0.8

        colors = dict(
            observed="#009E73",
            telluric="#E69F00",
            corrected="#111827",
            halo="#000000",
            marker="#64748b",
        )

    return colors


def style_legend(p, base: str) -> None:
    if not p.legend:
        return

    if base == "dark":
        fg = "#e5e7eb"
        bg = "#0b1220"
        border = "#334155"
        alpha = 0.35
    else:
        fg = "#111827"
        bg = "white"
        border = "#d1d5db"
        alpha = 0.90

    for leg in p.legend:
        leg.label_text_color = fg
        leg.title_text_color = fg
        leg.label_text_font_size = "11pt"
        leg.background_fill_color = bg
        leg.background_fill_alpha = alpha
        leg.border_line_color = border
        leg.border_line_alpha = 0.9


def line_with_halo(p, x, y, color, halo_color, label=None, *, base="light"):
    if base == "dark":
        p.line(x, y, line_width=5.0, color=halo_color, alpha=0.35)
        return p.line(x, y, line_width=2.8, color=color, alpha=0.97, legend_label=label)
    else:
        return p.line(x, y, line_width=2.2, color=color, alpha=0.97, legend_label=label)


# ----------------------------
# Load data
# ----------------------------
with st.spinner("Loading data..."):
    time.sleep(0.05)

    for pth in (WV_FILE, OBS_FILE, ATM_FILE):
        if not pth.exists():
            st.error(f"Missing required file: {pth.resolve()}")
            st.stop()

    wvAng, fts_obs, fts_atm, fts_cor = load_atlas(str(WV_FILE), str(OBS_FILE), str(ATM_FILE))


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Solar Atlas Plotting")

theme_choice = st.sidebar.radio("Plot theme", ["Auto", "Light", "Dark"], index=0, horizontal=True)
theme_base = resolve_theme_base(theme_choice)

wvCen = st.sidebar.number_input(
    "Center Wavelength [Angstrom]",
    min_value=float(wvAng.min()),
    max_value=float(wvAng.max()),
    value=10830.0,
    step=1.0,
)

maxRange = float(min(wvCen - float(wvAng.min()), float(wvAng.max()) - wvCen))
wvRange = st.sidebar.number_input(
    "Range [Angstrom]",
    min_value=1.0,
    max_value=maxRange,
    value=10.0,
    step=1.0,
)

show_lines = st.sidebar.toggle("Show NIST atomic lines", value=False)
topN = st.sidebar.slider("Top N lines", 5, 80, 20, 1)

with st.sidebar.expander("NIST ranking controls"):
    max_Ei_eV = st.slider("Max lower-level energy Ei (eV)", 2.0, 30.0, 25.0, 0.5)
    prefer_neutral = st.toggle("Prefer neutral stage (I)", value=True)
    T_rank = st.slider("Ranking temperature proxy (K)", 3500, 8000, 5770, 100)

st.sidebar.markdown(
    """
## About this app
Used for plotting different spectral atlases relative to optical/infrared observations of the Sun.

Selected range is interpolated to 25000 points for memory reasons.

PRELIMINARY -- CHECK RESULTS! 

## Data Sources
NSO FTS

## Creator
Tom Schad — www.github.com/tschad
"""
)

st.markdown("# Solar Spectrum Plotter")


# ----------------------------
# Plot
# ----------------------------
pan = PanTool(dimensions="width")
wheel = WheelZoomTool(dimensions="width")
boxx = BoxZoomTool(dimensions="width")
reset = ResetTool()
save = SaveTool()

p = figure(
    sizing_mode="scale_width",
    aspect_ratio=1.6,
    height=300,
    min_height=250,
    max_height=800,
    title="Solar Spectrum",
    x_axis_label="Wavelength [angstrom]",
    y_axis_label="Intensity (normalized)",
    tools=[pan, wheel, boxx, reset, save],
    output_backend="webgl",
)

p.toolbar.active_drag = boxx
p.toolbar.active_scroll = wheel

colors = apply_axes_and_grid_style(p, theme_base)

wv_lo = wvCen - wvRange / 2.0
wv_hi = wvCen + wvRange / 2.0

npts = 25000
wvInt = np.linspace(wv_lo, wv_hi, int(npts))

y_obs = np.interp(wvInt, wvAng, fts_obs)
y_atm = np.interp(wvInt, wvAng, fts_atm)
y_cor = np.interp(wvInt, wvAng, fts_cor)

obs_src = ColumnDataSource(data={"x": wvInt, "y": y_obs})

# Dedicated marker y-range so markers don't distort autoscaling
p.extra_y_ranges = {"marker": Range1d(0, 1)}
p.add_layout(LinearAxis(y_range_name="marker", visible=False), "right")


# ----------------------------
# NIST atomic lines
# ----------------------------
if show_lines:
    with st.spinner("Querying NIST (this can take ~10–30 s)..."):
        tab_raw, nist_errors = nist_query_window_airish(float(wv_lo), float(wv_hi), tuple(SPECIES))

    tab_ranked = tighten_and_rank_nist_lines(
        tab_raw,
        float(wv_lo),
        float(wv_hi),
        T=float(T_rank),
        max_Ei_eV=float(max_Ei_eV),
        prefer_neutral=bool(prefer_neutral),
    )

    if tab_ranked is None or len(tab_ranked) == 0:
        st.info("No NIST atomic lines after tightening/ranking.")
    else:
        
        tab_top = tab_ranked[:topN]

        # Display table (keep it compact)
        cols = ["species", "wA", "wsrc"]
        for c in ["Observed", "Ritz", "Rel.", "Aki", "log_gf", "Ei", "Acc.", "score"]:
            if c in tab_top.colnames and c not in cols:
                cols.append(c)

        st.subheader("Top atomic lines (NIST ASD) — tightened ranking")
        st.dataframe(tab_top[cols], use_container_width=True)

        # Markers
        wA_top = np.array(tab_top["wA"], dtype=float)

        labels = []
        for i in range(len(tab_top)):
            sp = str(tab_top["species"][i])
            wl = float(tab_top["wA"][i])
            labels.append(f"{sp} {wl:.3f} Å")

        mark_src = ColumnDataSource(
            data=dict(
                x=wA_top,
                y0=np.full_like(wA_top, 0.90, dtype=float),
                y1=np.full_like(wA_top, 0.99, dtype=float),
                label=labels,
            )
        )

        markers = p.segment(
            x0="x", y0="y0", x1="x", y1="y1",
            source=mark_src,
            y_range_name="marker",
            line_width=2,
            line_alpha=0.75,
            line_color=colors["marker"],
        )

        p.add_tools(HoverTool(renderers=[markers], tooltips=[("Line", "@label")], mode="mouse"))


# ----------------------------
# Curves + hover (observed only)
# ----------------------------
if theme_base == "dark":
    p.line("x", "y", source=obs_src, line_width=5.0, color=colors["halo"], alpha=0.35)
    observed_renderer = p.line(
        "x", "y", source=obs_src,
        line_width=2.8, color=colors["observed"], alpha=0.97,
        legend_label="Observed Spectrum",
    )
else:
    observed_renderer = p.line(
        "x", "y", source=obs_src,
        line_width=2.2, color=colors["observed"], alpha=0.97,
        legend_label="Observed Spectrum",
    )

line_with_halo(p, wvInt, y_atm, colors["telluric"], colors["halo"], "Telluric", base=theme_base)
line_with_halo(p, wvInt, y_cor, colors["corrected"], colors["halo"], "Corrected", base=theme_base)

hover = HoverTool(
    renderers=[observed_renderer],
    tooltips=[("λ [Å]", "@x{0.000}"), ("I_obs", "@y{0.0000}")],
    mode="vline",
)
p.add_tools(hover)

p.legend.location = "top_right"
p.legend.click_policy = "hide"
style_legend(p, theme_base)

streamlit_bokeh(
    p,
    use_container_width=True,
    theme="streamlit",
    key="solar_spectrum_bokeh",
)
