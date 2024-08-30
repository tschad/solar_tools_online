import streamlit as st

def main():
    st.title("ViSP IPC")

    st.header("Configuration")
    config_name = st.text_input("Name:", value="-")
    
    mode = st.selectbox("Mode:", ["Mode1", "Mode2", "Mode3"])
    
    grating = st.selectbox("Grating:", ["Grating1", "Grating2", "Grating3"])
    
    tilt_angle = st.number_input("Tilt Angle:", value=-1.0)
    
    slit = st.selectbox("Slit:", ["Slit1", "Slit2", "Slit3"])
    
    if st.button("Update"):
        st.write("Update button clicked")
    
    if st.button("Close"):
        st.write("Close button clicked")
    
    if st.button("Load Config"):
        st.write("Load Config button clicked")
    
    if st.button("Save Config"):
        st.write("Save Config button clicked")
    
    if st.button("Optimizer"):
        st.write("Optimizer button clicked")
    
    if st.button("IP/DSP Export"):
        st.write("IP/DSP Export button clicked")
    
    optimizer_output = st.selectbox("Load Optimizer Output:", ["N/A"])
    
    if st.button("Load Optimizer Output"):
        st.write("Load Optimizer Output button clicked")
    
    # Placeholder for arms base components
    st.subheader("Arms Base 1")
    st.subheader("Arms Base 2")
    st.subheader("Arms Base 3")

if __name__ == "__main__":
    main()
