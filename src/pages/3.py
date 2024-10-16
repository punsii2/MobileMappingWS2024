import os
from math import atan2, cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import streamlit as st
import transforms3d as tr
from plyfile import PlyData
from streamlit_back_camera_input import back_camera_input

EXERCISE = 3
TASKS = 4
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))
task = 0

task += 1
with tabs[task - 1]:
    image = back_camera_input()
    if image:
        st.image(image)

    st.write(f"## 7.1.{EXERCISE}.{task}")

    with st.sidebar:
        st.write("sidebar")
    st.write("content")
