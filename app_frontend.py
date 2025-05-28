import streamlit as st
from ocp_solver import solve_ocp
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="Optimal Flight Trajectory", layout="centered")
st.title("Optimal Flight Trajectory Solver")
st.caption("Abort Landing under Wind Shear")

st.markdown("Please input the parameter for wind strength **k**, and press the button to generate the abort landing trajectory:")

k = st.slider("wind strength (default setting is 1.0)", min_value=0.0, max_value=2.0, value=1.0, step=0.01)

if st.button("Find solution"):
    with st.spinner("Computing..."):
        try:
            result = solve_ocp(k)
            st.success("Solution found")
            img_bytes = base64.b64decode(result["image_base64"])
            image = Image.open(io.BytesIO(img_bytes))
            st.image(image, caption="Optimal Trajectory", use_column_width=True)

        except Exception as e:
            st.error(f"Fail to solve: {e}")