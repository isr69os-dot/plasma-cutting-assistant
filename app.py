import streamlit as st

st.set_page_config(page_title="Plasma Assistant", page_icon="🔥")

st.title("🔥 Plasma Cutting Assistant")

st.write("First local test is working.")

material = st.selectbox(
    "Select material",
    ["Black Steel", "Galvanized Steel", "Stainless", "Aluminum"]
)

thickness = st.number_input(
    "Thickness [mm]",
    min_value=0.5,
    max_value=30.0,
    value=3.0,
    step=0.5
)

st.write(f"Selected: {material} | {thickness} mm")