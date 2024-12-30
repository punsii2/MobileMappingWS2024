import os
from math import pi
from pathlib import Path

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import transforms3d as tr
from numpy.linalg import inv

from utils.plot import (
    SH,
    SW,
    init_figure,
    plot_camera,
    plot_points,
    plot_world_coordinates,
)

EXERCISE = 5
TASKS = 2
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))


task = 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    DATA_PATH = Path(os.getcwd() + "/../data/LV_5/")
    LEFT_PATH = DATA_PATH / "scene_l.jpg"
    RIGHT_PATH = DATA_PATH / "scene_r.jpg"

    # read an image from disk
    image_left = cv.cvtColor(cv.imread(str(LEFT_PATH)), cv.COLOR_BGR2GRAY)
    image_right = cv.cvtColor(cv.imread(str(RIGHT_PATH)), cv.COLOR_BGR2GRAY)
    cl, cr = st.columns(2)
    cl.image(image_left)
    cr.image(image_right)

    anaglyph = np.stack([image_left, image_right, image_right], axis=2)

    # Initialize the stereo block matching object
    stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=15)
    # Compute the disparity image
    disparity = stereo.compute(image_left, image_right)
    dl, dr = st.columns(2)
    dl.image(anaglyph)
    dr.image(disparity.astype(np.uint8))

    disparity_vector = disparity.ravel()
    # in stereogemoetry disparity < 0 are not valid
    filter_mask_disparity = disparity_vector > 0
    disparity_vector = disparity_vector[filter_mask_disparity]
    st.write("Histogram of disparity map:")
    hist = go.Figure()
    hist.add_trace(
        go.Histogram(
            x=disparity_vector,
            opacity=0.5,
            # marker_color=color,
            # name="%s channel" % color,
        )
    )
    st.plotly_chart(hist)

    st.divider()
    FOCAL_LENGTH = 0.08 * 2000
    CAMERA_DIST = 0.4 * 2000

    w, h = image_left.shape
    image_points = []
    for x in range(w):
        for y in range(h):
            image_points.append((x - int(w / 2), y - int(h / 2), image_left[x][y]))
    image_points = np.array(image_points)[filter_mask_disparity]

    zr = np.array(FOCAL_LENGTH * CAMERA_DIST / -disparity_vector)
    xr = np.multiply(zr, image_points[:, 0] / FOCAL_LENGTH)
    yr = np.multiply(zr, image_points[:, 1] / FOCAL_LENGTH)

    filter_mask_result = (zr > -2200) & (zr < -400)
    zr = zr[filter_mask_result]
    xr = xr[filter_mask_result]
    yr = yr[filter_mask_result]

    point_cloud = init_figure(
        eye={"x": 0, "y": 0, "z": 1.0}, up={"x": 1, "y": 0, "z": 0}
    )
    points_reconstructed = np.stack((xr.ravel(), yr.ravel(), zr.ravel()), axis=1)
    # In theory we still need to go back to world coordinates
    # points_reconstructed = points_reconstructed @ R1.T + T1
    point_cloud.add_trace(
        go.Scatter3d(
            x=xr,
            y=yr,
            z=zr,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(
                size=3,
                color=image_left.ravel()[filter_mask_disparity][filter_mask_result],
                line_width=0.0,
                colorscale="gray",
            ),
        )
    )
    st.plotly_chart(point_cloud)


task = 2
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    st.write(f"# Color the disparity ")

    xmin, xmax, ymin, ymax = -5, 5, -5, 5

    R1 = tr.euler.euler2mat(0.5 * pi, 0, -0.25 * pi)
    T1 = [xmax, ymax, 0]
    W1 = tr.affines.compose(T1, R1, np.ones(3))
    # R1 = tr.euler.euler2mat(pi, 0, 0)
    # T1 = [0, 0, 5]
    # W1 = tr.affines.compose(T1, R1, np.ones(3))

    R2 = tr.euler.euler2mat(0.5 * pi, 0, 0.25 * pi)
    T2 = [xmin, ymax, 0]
    W2 = tr.affines.compose(T2, R2, np.ones(3))

    fig1 = init_figure(
        size=800, eye={"x": 1.0, "y": 1.0, "z": 0.5}, up={"x": 0, "y": 0, "z": 1}
    )
    plot_world_coordinates(
        fig1, xrange=[xmin, xmax], yrange=[ymin, ymax], zrange=[0, 4]
    )

    plot_camera(fig1, W1, name="Camera 1", color="red")
    plot_camera(fig1, W2, name="Camera 2", color="blue")

    fig1.add_trace(
        go.Scatter3d(
            x=[xmin, xmax],
            y=[ymin, ymax],
            z=[0, 0],
            mode="lines",
            legendgroup="Optical axis red camera",
            name="Camera",
            line=dict(color="black", width=5),
            showlegend=False,
            hovertemplate="",
        )
    )
    fig1.add_trace(
        go.Scatter3d(
            x=[xmin, xmax],
            y=[ymax, ymin],
            z=[0, 0],
            mode="lines",
            legendgroup="Optical axis blue camera",
            name="Camera",
            line=dict(color="black", width=5),
            showlegend=False,
            hovertemplate="",
        )
    )

    xy = (
        np.mgrid[
            1.0 * xmin + 0.1 : xmax - 0.25 : 0.1,
            1.0 * ymin + 0.1 : ymax - 1.25 : 0.1,
        ]
        .reshape(2, -1)
        .T
    )
    n = xy.shape[0]
    RP = tr.euler.euler2mat(0, 0, -0.25 * pi)
    p0 = np.c_[xy, np.zeros(n)]

    plot_base_plane = st.checkbox("Plot base plane")
    use_disparity_as_z = False
    if not plot_base_plane:
        use_disparity_as_z = st.checkbox("Elevate result by disparity")

    if plot_base_plane:
        use_disparity_as_z = True
        plot_points(fig1, p0, "green", name="grid samples")

    p1 = (p0 - T1) @ inv(R1.T)
    p2 = (p0 - T2) @ inv(R2.T)

    b1 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], p1)))
    b2 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], p2)))
    disparity = np.multiply(np.subtract(b2[:, 0], b1[:, 0]), 2)

    x, y, z = p0.T
    z = disparity if use_disparity_as_z else z
    fig1.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(size=5, color=-disparity, line_width=0.0, colorscale="RdBu"),
        )
    )

    c1_disparity = (np.array(list(zip(x, y, z))) - T1) @ inv(R1.T)
    c1_disparity_projected = np.array(
        list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], c1_disparity))
    )
    c1xp, c1yp, c1zp = c1_disparity_projected.T
    mask = (-SW <= c1xp) * (c1xp <= SW) * (-SH <= c1yp) * (c1yp <= SH)
    c1_disparity_projected_filtered = c1_disparity_projected[mask, :]
    c1xf, c1yf, c1zf = (c1_disparity_projected_filtered @ R1.T + T1).T
    fig1.add_trace(
        go.Scatter3d(
            x=c1xf,
            y=c1yf,
            z=c1zf,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(size=1, color=-c1zf, line_width=0.0, colorscale="RdBu"),
        )
    )

    c2_disparity = (np.array(list(zip(x, y, z))) - T2) @ inv(R2.T)
    c2_disparity_projected = np.array(
        list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], c2_disparity))
    )
    c2xp, c2yp, c2zp = c2_disparity_projected.T
    mask = (-SW <= c2xp) * (c2xp <= SW) * (-SH <= c2yp) * (c2yp <= SH)
    c2_disparity_projected_filtered = c2_disparity_projected[mask, :]
    c2xf, c2yf, c2zf = (c2_disparity_projected_filtered @ R2.T + T2).T
    fig1.add_trace(
        go.Scatter3d(
            x=c2xf,
            y=c2yf,
            z=c2zf,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(size=1, color=-c2zf, line_width=0.0, colorscale="RdBu"),
        )
    )

    xmin, xmax, ymin, ymax = -5, 5, -5, 15

    R1 = tr.euler.euler2mat(0.5 * pi, 0, 0)
    T1 = [xmax - 1, ymax, 0]
    W1 = tr.affines.compose(T1, R1, np.ones(3))
    # R1 = tr.euler.euler2mat(pi, 0, 0)
    # T1 = [0, 0, 5]
    # W1 = tr.affines.compose(T1, R1, np.ones(3))

    R2 = tr.euler.euler2mat(0.5 * pi, 0, 0)
    T2 = [xmin + 1, ymax, 0]
    W2 = tr.affines.compose(T2, R2, np.ones(3))

    fig2 = init_figure(
        size=800, eye={"x": 1.0, "y": 1.0, "z": 0.5}, up={"x": 0, "y": 0, "z": 1}
    )
    plot_world_coordinates(
        fig2, xrange=[xmin, xmax], yrange=[ymin, ymax], zrange=[0, 4]
    )

    plot_camera(fig2, W1, name="Camera 1", color="red")
    plot_camera(fig2, W2, name="Camera 2", color="blue")

    fig2.add_trace(
        go.Scatter3d(
            x=[xmin, xmax],
            y=[ymin, ymax],
            z=[0, 0],
            mode="lines",
            legendgroup="Optical axis red camera",
            name="Camera",
            line=dict(color="black", width=5),
            showlegend=False,
            hovertemplate="",
        )
    )
    fig2.add_trace(
        go.Scatter3d(
            x=[xmin, xmax],
            y=[ymax, ymin],
            z=[0, 0],
            mode="lines",
            legendgroup="Optical axis blue camera",
            name="Camera",
            line=dict(color="black", width=5),
            showlegend=False,
            hovertemplate="",
        )
    )

    xy = (
        np.mgrid[
            1.0 * xmin + 0.1 : xmax - 0.25 : 0.1,
            1.0 * ymin + 0.1 : ymax - 1.25 : 0.1,
        ]
        .reshape(2, -1)
        .T
    )
    n = xy.shape[0]
    RP = tr.euler.euler2mat(0, 0, -0.25 * pi)
    p0 = np.c_[xy, np.zeros(n)]
    if plot_base_plane:
        use_disparity_as_z = True
        plot_points(fig2, p0, "green", name="grid samples")

    p1 = (p0 - T1) @ inv(R1.T)
    p2 = (p0 - T2) @ inv(R2.T)

    b1 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], p1)))
    b2 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], p2)))
    disparity = np.multiply(np.subtract(b2[:, 0], b1[:, 0]), 2)

    x, y, z = p0.T
    z = disparity if use_disparity_as_z else z
    fig2.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(size=5, color=-disparity, line_width=0.0, colorscale="RdBu"),
        )
    )

    c1_disparity = (np.array(list(zip(x, y, z))) - T1) @ inv(R1.T)
    c1_disparity_projected = np.array(
        list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], c1_disparity))
    )
    c1xp, c1yp, c1zp = c1_disparity_projected.T
    mask = (-SW <= c1xp) * (c1xp <= SW) * (-SH <= c1yp) * (c1yp <= SH)
    c1_disparity_projected_filtered = c1_disparity_projected[mask, :]
    c1xf, c1yf, c1zf = (c1_disparity_projected_filtered @ R1.T + T1).T
    fig2.add_trace(
        go.Scatter3d(
            x=c1xf,
            y=c1yf,
            z=c1zf,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(size=1, color=-c1zf, line_width=0.0, colorscale="RdBu"),
        )
    )

    c2_disparity = (np.array(list(zip(x, y, z))) - T2) @ inv(R2.T)
    c2_disparity_projected = np.array(
        list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], c2_disparity))
    )
    c2xp, c2yp, c2zp = c2_disparity_projected.T
    mask = (-SW <= c2xp) * (c2xp <= SW) * (-SH <= c2yp) * (c2yp <= SH)
    c2_disparity_projected_filtered = c2_disparity_projected[mask, :]
    c2xf, c2yf, c2zf = (c2_disparity_projected_filtered @ R2.T + T2).T
    fig2.add_trace(
        go.Scatter3d(
            x=c2xf,
            y=c2yf,
            z=c2zf,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(size=1, color=-c2zf, line_width=0.0, colorscale="RdBu"),
        )
    )

    disparity_columns = st.columns(2)
    disparity_columns[0].plotly_chart(fig1)
    disparity_columns[1].plotly_chart(fig2)
