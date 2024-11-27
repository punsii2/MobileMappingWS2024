import numpy as np
import plotly.graph_objects as go
import streamlit as st
import transforms3d as tr
from numpy.linalg import inv

from utils.plot import init_figure, plot_camera, plot_points, plot_world_coordinates

EXERCISE = 7
TASKS = 1
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)
task = 0

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))

task += 1
st.write(f"## 7.1.{EXERCISE}.{task}")

W = tr.affines.compose(np.zeros(3), np.identity(3), np.ones(3))

fig1 = init_figure()
plot_world_coordinates(fig1, xrange=[-3, 3], yrange=[-3, 3], zrange=[0, 4])

plot_camera(fig1, W, name="World", color="black")

# create Camera K1 coordinate system
T1 = [-0.5, 0, 1]
R1 = tr.euler.euler2mat(0.1, 0.4, 0)
K1 = tr.affines.compose(T1, R1, np.ones(3))
plot_camera(fig1, K1, name="K1", color="red")

# create Camera K2 coordinate system
T2 = [0.5, 0, 1]
R2 = tr.euler.euler2mat(-0.1, -0.4, 0)
K2 = tr.affines.compose(T2, R2, np.ones(3))
plot_camera(fig1, K2, name="K2", color="blue")


# define camera parameters
cx1, cy1 = 500, 350
fx1, fy1 = 700, 700
cx2, cy2 = 500, 350
fx2, fy2 = 700, 700
M1 = np.array(
    [
        [fx1, 0, cx1],
        [0, fy1, cy1],
        [0, 0, 1],
    ]
)
M2 = np.array(
    [
        [fx2, 0, cx2],
        [0, fy2, cy2],
        [0, 0, 1],
    ]
)

T = inv(K1) @ K2

# create line plots originating in the K2 coordinate system
linspace = np.linspace(0, 4, 50)
lines = [
    [0, 0, 1],
    [0.1, 0.5, 1],
    [-0.3, 0.5, 1],
    [0.1, -0.5, 1],
]
PointsK1 = []
for line in lines:
    PointsK1.extend(
        np.array(
            list(
                map(
                    lambda s: [s * line[0], s * line[1], s * line[2], 1],
                    linspace,
                )
            )
        )
    )
PointsK1 = np.array(PointsK1)
# transform points to the world coordinate system.
PointsW = PointsK1 @ K1.T
plot_points(fig1, PointsW, "black", name="points in world coordinate system")


# transform points from world coordinates to the K2 coordinate system
PointsK2 = PointsW @ inv(K2.T)
# project points to image plane
pointsK1 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], PointsK1)))
pointsK2 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], PointsK2)))
plot_points(fig1, pointsK2, "blue", name="points in K2 coordinate system")

st.plotly_chart(fig1)

# scale points in order to have pixel-values as coordinates
pixelsK1 = M1 @ pointsK1.T
pixelsK2 = M2 @ pointsK2.T
fig2 = init_figure()
fig2.update_layout(
    yaxis=dict(scaleanchor="x", scaleratio=1),
    height=1080 / 2,
    width=1920 / 2,
)
fig2.add_trace(
    go.Scatter(
        x=pixelsK2[0],
        y=pixelsK2[1],
        marker=dict(color="blue"),
        name="Points in image pixel coordinates",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[0, 0, 1920, 1920, 0],
        y=[0, 1080, 1080, 0, 0],
        marker=dict(color="red"),
        name="Viewport?",
        mode="lines",
    )
)
fig2.update_yaxes(autorange="reversed")

st.write("Resultig image (rendered from camera K2):")
st.plotly_chart(fig2)


B = np.array(T1) - np.array(T2)
S = [
    [0, -B[2], B[1]],
    [B[2], 0, -B[0]],
    [-B[1], B[0], 0],
]
R = inv(R1) @ R2
# st.write(PointsK2.T)
# st.write(T @ PointsK2.T)
# st.write(PointsK1.T) => same as above => correct Transform

E = R @ S
st.code(f"E = R @ S = \n{E}")
# st.write(PointsK1[:, 0:3] @ E @ PointsK2[:, 0:3].T)
# st.write(PointsK1[:, 0:3] @ E @ PointsK2[:, 0:3].T)

tmp = [13, 17, 1]
# st.write(inv(M1))
# st.write(M1 @ tmp)
# st.write(inv(M1) @ M1 @ tmp)
F = inv(M1).T @ E @ inv(M2)
st.code(f"F = inv(M1) @ E @ inv(M2)\n{F}")

st.code("pixelsK1.T @ F @ pixelsK2 = ")
st.write(pixelsK1.T @ F @ pixelsK2)
st.code(f"max(pixelsK1.T @ F @ pixelsK2.T) = {max((pixelsK1.T @ F @ pixelsK2)[1])}")
st.code(f"min(pixelsK1.T @ F @ pixelsK2.T) = {min((pixelsK1.T @ F @ pixelsK2)[1])}")
