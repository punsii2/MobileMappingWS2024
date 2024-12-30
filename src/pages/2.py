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

EXERCISE = 2
TASKS = 4
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))


# default parameters of a camera in Blender
SENSOR_SIZE = np.array([0.036, 0.024])
INTRINSIC_MATRIX = np.array(
    [[0.05, 0, SENSOR_SIZE[0] / 2.0], [0, 0.05, SENSOR_SIZE[1] / 2.0], [0, 0, 1]]
)
VIRTUAL_IMAGE_DISTANCE = 10

task = 1
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


task = 2
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    st.code(
        """
        def setRPY(r: float, p: float, y: float):
            cr, cp, cy = cos(r), cos(p), cos(y)
            sr, sp, sy = sin(r), sin(p), sin(y)
            return np.matrix(
                [
                    [cp * cy, cy * sr * sp - cr * sy, sr * sy + cr * cy * sp],
                    [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - cy * sr],
                    [-sp, cp * sr, cr * cp],
                ]
            )

        def getRPY(R):
            y = atan2(R[1, 0], R[0, 0])
            p = atan2(-R[2, 0], R[0, 0] * cos(y) + R[1, 0] * sin(y))
            r = atan2(R[2, 1] / cos(p), R[2, 2] / cos(p))
            return [r, p, y]
    """
    )

    def setRPY(r: float, p: float, y: float):
        cr, cp, cy = cos(r), cos(p), cos(y)
        sr, sp, sy = sin(r), sin(p), sin(y)
        return np.matrix(
            [
                [cp * cy, cy * sr * sp - cr * sy, sr * sy + cr * cy * sp],
                [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - cy * sr],
                [-sp, cp * sr, cr * cp],
            ]
        )

    def getRPY(R):
        y = atan2(R[1, 0], R[0, 0])
        p = atan2(-R[2, 0], R[0, 0] * cos(y) + R[1, 0] * sin(y))
        r = atan2(R[2, 1] / cos(p), R[2, 2] / cos(p))
        return [r, p, y]

    [r, p, y] = angles
    st.code(f"{[r, p, y]=}")

    M = setRPY(r, p, y)
    st.code("M = setRPY(r, p, y) = ")
    st.write(pd.DataFrame(M))

    [r2, p2, y2] = getRPY(M)
    st.code("[r2, p2, y2] = getRPY(M) = ")
    st.write(getRPY(M))


task = 3
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    def isRot(M):
        # allclose handles floats more gracefully than equals
        return np.allclose(np.matmul(M, M.T), np.identity(3)) and np.allclose(
            np.linalg.det(M), 1
        )

    st.code(
        """def isRot(M):
    # allclose handles floats more gracefully than equals
    return (
        np.allclose(np.matmul(M, M.T), np.identity(3))
        and np.allclose(np.linalg.det(M), 1.0)
    )
    """
    )

    st.write("M = ")
    st.write(pd.DataFrame(M))
    st.code(f"isRot(M) = {isRot(M)}")

task = 4
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    M4 = [
        [cos(pi / 4.0), sin(pi / 4.0), 0.0, 0.0],
        [-sin(pi / 4.0), cos(pi / 4.0), 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    [T, R, _, _] = tr.affines.decompose44(M4)
    st.code(f"{M4 = }")
    st.code(f"{isRot(R) = }")
    st.code(f"{getRPY(R) = }")

    DATA_PATH = Path(os.getcwd() + "/../data/LV_2")
    plydata = PlyData.read(DATA_PATH / "teapotOut.ply")
    points = plydata["vertex"]

    df = pd.DataFrame()
    df["x"] = points["x"]
    df["y"] = points["y"]
    df["z"] = points["z"]

    st.write("Original teapot")
    fig = px.scatter_3d(df, x="x", y="y", z="z")
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Use manual transformation"):
        T = t
        R = tr.euler.euler2mat(angles[0], angles[1], angles[2])
    else:
        st.write("(Using M4...)")
    st.write("Transformed teapot")

    df.iloc[:, :] += T
    df = df.dot(R)

    fig = px.scatter_3d(df, x=0, y=1, z=2)
    st.plotly_chart(fig, use_container_width=True)
