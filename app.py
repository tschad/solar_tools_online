# http://awesome-streamlit.org/
# https://blog.esciencecenter.nl/forget-about-jupyter-notebooks-showcase-your-research-using-dashboards-5d13451ba374

# Also works with most supported chart types
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time
from bokeh.plotting import figure
from bokeh.models import HoverTool

st.set_page_config(layout="wide")

with st.spinner('Loading data...'):
    time.sleep(0.2)
    wvAng   = np.load('./combined_fts_v5June2022_wavelength_angstrom.npy').astype(float)
    fts_obs = np.load('./combined_fts_v5June2022_observed_spectrum.npy').astype(float)
    fts_atm = np.load('./combined_fts_v5June2022_atm_absorption.npy').astype(float)
    fts_cor = np.clip(fts_obs/fts_atm,0.,1.05)

st.sidebar.header("Solar Atlas Plotting")

wvCen = st.sidebar.number_input(r"Center Wavelength [Angstrom]",min_value = wvAng.min(),max_value = wvAng.max(),value = 6563.,step=1.)
maxRange = np.min(np.array([wvCen - wvAng.min(),wvAng.max()-wvCen]))
wvRange = st.sidebar.number_input(r"Range [Angstrom]",min_value = 1.,max_value = maxRange,value = 10.,step = 1.)

st.sidebar.markdown(
"""
## About this app

Used for plotting different spectral atlases relative to optical/infrared observations of the Sun.

## Data Sources 

NSO FTS 

## Creator 

Tom Schad [www.github.com/tschad]

"""
)

st.markdown(
    """
# Solar Spectrum Plotter

[Bokeh](https://docs.bokeh.org/en/latest/) is an interactive visualization library for modern web
browsers. Bokeh has an extensive
[quickstart guide](https://docs.bokeh.org/en/latest/docs/user_guide/quickstart.html), a large
[gallery of examples](https://docs.bokeh.org/en/latest/docs/gallery.html#gallery) and an extensive
[reference guide](https://docs.bokeh.org/en/latest/docs/reference.html#refguide)

Temporary.  The FTS spectrum 
"""
)

###-----------

p = figure(sizing_mode="stretch_width", height=400,
    title = 'Solar Spectrum',
    x_axis_label=r"Wavelength [angstrom]",
    y_axis_label='Intensity (normalized)') 

wvInt = np.linspace(wvCen-wvRange/2.,wvCen+wvRange/2.,10000)

p.line(wvInt,np.interp(wvInt,wvAng,fts_obs),legend_label='Observed', line_width=2,color = 'green')
p.line(wvInt,np.interp(wvInt,wvAng,fts_atm),legend_label='Telluric', line_width=2,color = 'orange')
p.line(wvInt,np.interp(wvInt,wvAng,fts_cor) ,legend_label='Corrected', line_width=2,color = 'black')

p.add_tools(HoverTool(tooltips="y: @y, x: @x", mode="vline"))

st.bokeh_chart(p) #use_container_width=True)
