import os
from math import pi
from pathlib import Path
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytransform3d.camera as pc
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import streamlit as st
import transforms3d as tr
from plyfile import PlyData

EXERCISE = 2
TASKS = 4
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))
task = 0


# default parameters of a camera in Blender
SENSOR_SIZE = np.array([0.036, 0.024])
INTRINSIC_MATRIX = np.array(
    [[0.05, 0, SENSOR_SIZE[0] / 2.0], [0, 0.05, SENSOR_SIZE[1] / 2.0], [0, 0, 1]]
)
VIRTUAL_IMAGE_DISTANCE = 10

task += 1
with tabs[task - 1]:

    st.write(f"## 7.1.{EXERCISE}.{task}")

    with st.sidebar:
        st.write("Rotation:")
        angles = [
            st.slider("rx", -pi, pi, 0.0),
            st.slider("ry", -pi, pi, 0.0),
            st.slider("rz", -pi, pi, 0.0),
        ]
        st.divider()
        st.write("Translation:")
        t = [
            st.slider("tx", -10.0, 10.0, 0.0),
            st.slider("ty", -10.0, 10.0, 0.0),
            st.slider("tz", -10.0, 10.0, 0.0),
        ]

    [c1, c2, c3] = st.columns(3)
    with c1:
        rotation = tr.euler.euler2mat(angles[0], angles[1], angles[2])
        st.write("Rotation matrix:", rotation)

    with c2:
        transformRotation = tr.affines.compose(np.zeros(3), rotation, np.ones(3))
        st.write("Homogenous rotation matrix:", transformRotation)

    with c3:
        transformTranslation = tr.affines.compose(t, np.identity(3), np.ones(3))
        st.write("Homogenous translation matrix:", transformTranslation)

    st.divider()

    [c1, c2, c3] = st.columns(3)
    with c1:
        rotTrans = np.matmul(transformRotation, transformTranslation)
        st.write("Multiplied transforms (rotation first):", rotTrans)

    with c2:
        K1 = np.matmul(transformTranslation, transformRotation)
        st.write("Multiplied transforms (translation first):", K1)

    st.divider()
    st.warning("The X and Y axes are swapped compared to the matlab plots.", icon="🚨")

    camera_plot = plt.figure()
    ax = pt.plot_transform(s=2, ax_s=10)
    pc.plot_camera(
        ax,
        ax_s=10,
        cam2world=K1,
        M=INTRINSIC_MATRIX,
        sensor_size=SENSOR_SIZE,
        virtual_image_distance=VIRTUAL_IMAGE_DISTANCE,
        color="yellow",
    )

    T12 = tr.affines.compose([0, 0, 5], tr.euler.euler2mat(-0.5 * pi, 0, 0), np.ones(3))
    K2 = np.matmul(K1, T12)
    pc.plot_camera(
        ax,
        ax_s=10,
        cam2world=K2,
        M=INTRINSIC_MATRIX,
        sensor_size=SENSOR_SIZE,
        virtual_image_distance=VIRTUAL_IMAGE_DISTANCE,
        color="green",
    )

    T13 = tr.affines.compose([0, 5, 0], tr.euler.euler2mat(0.5 * pi, 0, 0), np.ones(3))
    K3 = np.matmul(K1, T13)
    pc.plot_camera(
        ax,
        ax_s=10,
        cam2world=K3,
        M=INTRINSIC_MATRIX,
        sensor_size=SENSOR_SIZE,
        virtual_image_distance=VIRTUAL_IMAGE_DISTANCE,
        color="blue",
    )

    T23 = np.matmul(np.linalg.inv(T12), T13)
    K4 = np.matmul(K2, T23)
    pc.plot_camera(
        ax,
        ax_s=10,
        cam2world=K4,
        M=INTRINSIC_MATRIX,
        sensor_size=SENSOR_SIZE,
        virtual_image_distance=VIRTUAL_IMAGE_DISTANCE,
        color="red",  # => This is drawn over the camera in K3 => blue should be replaced by red
    )
    st.pyplot(camera_plot)

    [T, R, _, _] = tr.affines.decompose44(T23)
    txyz = tr.euler.mat2euler(T23)
    [c1, c2, c3] = st.columns(3)
    c1.write("Extracted tranlation from T23:")
    c1.write(T)
    c2.write("Extracted rotation from T23:")
    c2.write(R)
    c3.write("Extracted euler rotations from T23:")
    c3.write(txyz)


task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
