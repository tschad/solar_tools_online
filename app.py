
# http://awesome-streamlit.org/
# https://blog.esciencecenter.nl/forget-about-jupyter-notebooks-showcase-your-research-using-dashboards-5d13451ba374

# Also works with most supported chart types
import numpy as np
import streamlit as st
import time
from bokeh.plotting import figure
from bokeh.models import HoverTool

st.set_page_config(layout="wide")

## LOAD SOLAR ATLAS
def vac2air(wave_vac):
    """ Converts wavelengths from vacuum to air-equivalent """
    wave_air = np.copy(wave_vac)
    ww  = (wave_vac >= 2000) 
    sigma2 = (1e4 / wave_vac[ww])**2 
    n = 1 + 0.0000834254 + 0.02406147 / (130 - sigma2) + 0.00015998 / (38.9 - sigma2)
    wave_air[ww] = wave_vac[ww] / n
    return wave_air

@st.cache_data
def load_telluric_atlas(): 
    ## Telluric synthetic atlas
    wvAng  = np.load('./telluric_atlas_mainMol_USstd_wv_air_angstrom_v20240307.npy').astype(float)
    spTell = np.load('./telluric_atlas_mainMol_USstd_CO2_416ppm-Base_3km-PWV_3__mm-Airmass_1___v20240307.npy').astype(float)
    return wvAng,spTell

@st.cache_data
def load_toon_atlas(): 
    ## Telluric synthetic atlas
    ## Disk Center TOON Spectra 
    dc = np.loadtxt('./solar_merged_20200720_600_33300_000.out',skiprows =3)  ## disk center
    dcwv = vac2air(1e7/dc[:,0]*10)[::-1]  #/ 10.
    dcsp = dc[:,1][::-1]  
    ## Disk Center TOON Spectra 
    dc = np.loadtxt('./solar_merged_20200720_600_33300_100.out',skiprows =3)  ## disk integ -- 100
    diwv = vac2air(1e7/dc[:,0]*10)[::-1]  #/ 10.
    disp = dc[:,1][::-1]  
    return dcwv,dcsp,diwv,disp 

@st.cache_data
def load_neckel_diskcen_atlas(): 
    dcen = np.load('./Neckel_Labs/neckel_labs_1984_disk_center_atlas.npy') 
    neckel_labs_wv_diskcent = dcen[:,0] 
    neckel_labs_Inorm_diskcenter = dcen[:,1]/dcen[:,2]
    return neckel_labs_wv_diskcent,neckel_labs_Inorm_diskcenter

@st.cache_data
def load_neckel_integ_atlas(): 
    dcen = np.load('./Neckel_Labs/neckel_labs_1984_disk_integated_atlas.npy')
    neckel_labs_wv_diskint = dcen[:,0] 
    neckel_labs_Inorm_diskint = dcen[:,1]/dcen[:,2]
    return neckel_labs_wv_diskint,neckel_labs_Inorm_diskint

with st.spinner('Loading data...'):
    time.sleep(0.2)
    wvAng,spTell = load_telluric_atlas()
    ## Disk Center TOON Spectra 
    dcwv,dcsp,diwv,disp  = load_toon_atlas()
    ## Neckel labs 
    neckel_labs_wv_diskcent,neckel_labs_Inorm_diskcenter= load_neckel_diskcen_atlas()    
    ## Neckel labs 
    neckel_labs_wv_diskint,neckel_labs_Inorm_diskint = load_neckel_integ_atlas()

st.sidebar.header("Solar Atlas Plotting")

#wvCen = st.sidebar.number_input(r"Center Wavelength [Angstrom]",min_value = wvAng.min(),max_value = wvAng.max(),value = 6302.,step=1.)
#maxRange = np.min(np.array([wvCen - wvAng.min(),wvAng.max()-wvCen]))
#wvRange = st.sidebar.number_input(r"Range [Angstrom]",min_value = 1.,max_value = maxRange,value = wvCen/1000.,step = 1.)

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
# Solar and Telluric Reference Spectra Plotter

Welcome.  This is a simple reference spectrum plotter for solar and telluric spectra. 

Telluric spectra from  https://github.com/tschad/dkist_telluric_atlas

"""
)

###-----------

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

wvCen = st.number_input(r"Center Wavelength [Angstrom]",min_value = wvAng.min(),max_value = wvAng.max(),value = 6302.,step=1.)
maxRange = np.min(np.array([wvCen - wvAng.min(),wvAng.max()-wvCen]))
wvRange = st.number_input(r"Range [Angstrom]",min_value = 1.,max_value = maxRange,value = wvCen/1000.,step = 1.)

p = figure(sizing_mode="stretch_width", height=400,
    title = 'Solar Spectrum',
    x_axis_label=r"Wavelength [angstrom]",
    y_axis_label='Intensity (normalized)') 

wvInt = np.linspace(wvCen-wvRange/2.,wvCen+wvRange/2.,10000)

p.line(wvInt,np.interp(wvInt,wvAng,spTell),legend_label='Synthetic Telluric', line_width=2,color = 'green')
p.line(wvInt,np.interp(wvInt,dcwv,dcsp),legend_label='Toon Disk Center Atlas', line_width=2,color = 'blue')
p.line(wvInt,np.interp(wvInt,diwv,disp),legend_label='Toon Disk Integ', line_width=2,color = 'black')
p.line(wvInt,np.interp(wvInt,neckel_labs_wv_diskcent,neckel_labs_Inorm_diskcenter),legend_label='Neckel Labs Disk Center', line_width=2,color = 'black')
p.line(wvInt,np.interp(wvInt,neckel_labs_wv_diskint,neckel_labs_Inorm_diskint),legend_label='Neckel Labs Disk Integrated', line_width=2,color = 'black')

p.add_tools(HoverTool(tooltips="y: @y, x: @x", mode="vline"))

st.bokeh_chart(p) #use_container_width=True)


left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
chosen = st.radio(
    'Atlas to plot',
    ("Toon Disk Center", "Toon Disk Integrated", "Hufflepuff", "Slytherin"))
st.write(f"You are in {chosen} house!")

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Disk Center Atlases", "Disk Integration Atlases"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

# You can also use "with" notation:
with tab1:
    st.radio("Select one:", [1, 2])
