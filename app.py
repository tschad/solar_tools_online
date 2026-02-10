import time
from pathlib import Path

import numpy as np
import streamlit as st
from bokeh.models import HoverTool, ColumnDataSource, BoxZoomTool, PanTool, WheelZoomTool, ResetTool, SaveTool
from bokeh.plotting import figure
from streamlit_bokeh import streamlit_bokeh  # pip install streamlit-bokeh

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
        raise ValueError(
            f"Expected 1D arrays; got shapes wv={wv.shape}, obs={obs.shape}, atm={atm.shape}"
        )
    if not (len(wv) == len(obs) == len(atm)):
        raise ValueError(f"Length mismatch: wv={len(wv)}, obs={len(obs)}, atm={len(atm)}")

    # Ensure monotonic wavelength for np.interp
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
            observed="#009E73",   # Okabe–Ito green
            telluric="#E69F00",   # Okabe–Ito orange
            corrected="#F8FAFC",  # near-white
            halo="#CBD5E1",       # light gray underlay
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
            halo="#000000",  # unused in light
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

st.sidebar.markdown(
    """
## About this app
Used for plotting different spectral atlases relative to optical/infrared observations of the Sun.

Selected range is interpolated to 25000 points for memory reasons

## Data Sources
NSO FTS

## Creator
Tom Schad — www.github.com/tschad
"""
)

st.markdown(
    """
# Solar Spectrum Plotter

TBD
"""
)


# ----------------------------
# Plot
# ----------------------------
# Explicit tools so we can make box-zoom x-only and set defaults
pan = PanTool(dimensions="width")
wheel = WheelZoomTool(dimensions="width")
boxx = BoxZoomTool(dimensions="width")  # x-only box zoom
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
)

# Make click-drag do x-only box zoom by default
p.toolbar.active_drag = boxx

# Keep wheel zoom active on scroll
p.toolbar.active_scroll = wheel

colors = apply_axes_and_grid_style(p, theme_base)

wv_lo = wvCen - wvRange / 2.0
wv_hi = wvCen + wvRange / 2.0

npts = 25000 

wvInt = np.linspace(wv_lo, wv_hi, int(npts))

y_obs = np.interp(wvInt, wvAng, fts_obs)
y_atm = np.interp(wvInt, wvAng, fts_atm)
y_cor = np.interp(wvInt, wvAng, fts_cor)

# Observed: ColumnDataSource so hover attaches only to this curve
obs_src = ColumnDataSource(data={"x": wvInt, "y": y_obs})

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

# Other curves (no hover attached)
line_with_halo(p, wvInt, y_atm, colors["telluric"], colors["halo"], "Telluric",  base=theme_base)
line_with_halo(p, wvInt, y_cor, colors["corrected"], colors["halo"], "Corrected", base=theme_base)

# Hover ONLY for observed curve
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
