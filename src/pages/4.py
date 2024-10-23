import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import transforms3d as tr

from utils.plot import init_figure, plot_camera, plot_world_coordinates

EXERCISE = 4
TASKS = 5
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)
task = 0

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    camera_plot = plt.figure()
    T = np.zeros(3)
    W = tr.affines.compose(T, np.identity(3), np.ones(3))
    T = [1, 2, 5]
    W = tr.affines.compose(T, np.identity(3), np.ones(3))

    fig = init_figure()
    plot_world_coordinates(fig)

    plot_camera(fig, W, name="World", color="black")

    T1 = [-0.5, 0, 1]
    K1 = tr.affines.compose(T1, np.identity(3), np.ones(3))
    plot_camera(fig, K1, name="K1", color="red")

    st.plotly_chart(fig)
