"""
3D visualization based on plotly.
1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go

# more genral calculation of camera vertices
# def camera_corners_base():
#     scale = 1
#     SENSOR_WIDTH = 0.036
#     SENSOR_HEIGHT = 0.024
#     CAMERA_INTRINSICS = np.array(
#         [[0.05, 0, SENSOR_WIDTH / 2.0], [0, 0.05, SENSOR_HEIGHT / 2.0], [0, 0, 1]]
#     )
#     corners = np.array(
#         [
#             [0, 0],
#             [SENSOR_WIDTH, 0],
#             [SENSOR_WIDTH, SENSOR_HEIGHT],
#             [0, SENSOR_HEIGHT],
#             [0, 0],
#         ]
#     )
#     return to_homogeneous(corners) @ np.linalg.inv(CAMERA_INTRINSICS).T * scale

# In our case however, hardcoding will do the trick
SCALE = 1
SW = 0.36  # sensor width
SH = 0.24  # sensor height
BASE_CAMERA_CORNERS = (
    np.array(
        [
            [-SW, -SH, 1.0],
            [SW, -SH, 1.0],
            [SW, SH, 1.0],
            [-SW, SH, 1.0],
            [-SW, -SH, 1.0],
        ]
    )
    * SCALE
)


def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(
    size: int = 1000, eye={"x": 0.5, "y": 1.0, "z": 0.7}, up={"x": 0, "y": 0, "z": 1}
) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=True,
        showbackground=True,
        showgrid=True,
        showline=True,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_white",
        height=size,
        width=size,
        scene_camera=dict(
            eye=eye,
            up=up,
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def plot_3d_bbox(
    fig: go.Figure,
    bbox: np.ndarray,  # size (8, 3)
    color: str = "rgba(255, 0, 0, 1)",
    width: int = 1,
):
    """Plot a set of 3D points.
      6 ------ 5
     / |     / |
    2 -|---- 1 |
    |  |     | |
    | 7 -----| 4
    |/   o   |/
    3 ------ 0
    """
    edges = [
        (0, 3),  # bottom
        (3, 7),
        (7, 4),
        (4, 0),
        (1, 2),  # top
        (2, 6),
        (6, 5),
        (5, 1),
        (0, 1),  # sides
        (3, 2),
        (7, 6),
        (4, 5),
    ]
    for start, end in edges:
        tr = go.Scatter3d(
            x=[bbox[start][0], bbox[end][0]],
            y=[bbox[start][1], bbox[end][1]],
            z=[bbox[start][2], bbox[end][2]],
            mode="lines",
            legendgroup=None,
            showlegend=False,
            line=dict(color=color, width=width),
        )
        fig.add_trace(tr)


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,  # size (n, 3)
    color: str = "black",
    ps: int = 4,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
    showlegend: Optional[bool] = True,
):
    """Plot a set of 3D points."""
    if pts.shape[1] == 3:
        x, y, z = pts.T
    else:
        x, y, z, _ = pts.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        marker=dict(size=ps, color=color, line_width=0.0, colorscale=colorscale),
    )
    fig.add_trace(tr)


def plot_camera(
    fig: go.Figure,
    W: np.ndarray,  # W, from camera to world, size (4,4)
    color: str = "black",
    name: Optional[str] = None,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    R = W[0:3, 0:3]
    t = W[0:3, 3]
    corners = BASE_CAMERA_CORNERS @ R.T + t

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=5),
        name=name,
        legendgroup=name,
        hovertemplate="Camera",
    )
    fig.add_trace(pyramid)

    xcords = [
        [0, 0, 0],
        [0.3, 0, 0],
    ] @ R.T + t
    x_axis = go.Scatter3d(
        x=xcords[:, 0],
        y=xcords[:, 1],
        z=xcords[:, 2],
        mode="lines",
        line=dict(color="blue", width=5),
        name=name,
        legendgroup=name,
        hovertemplate="X axis",
        showlegend=False,
    )
    ycords = [
        [0, 0, 0],
        [0, 0.3, 0],
    ] @ R.T + t
    y_axis = go.Scatter3d(
        x=ycords[:, 0],
        y=ycords[:, 1],
        z=ycords[:, 2],
        mode="lines",
        line=dict(color="red", width=5),
        name=name,
        legendgroup=name,
        hovertemplate="Y axis",
        showlegend=False,
    )
    zcords = [
        [0, 0, 0],
        [0, 0, 0.3],
    ] @ R.T + t
    z_axis = go.Scatter3d(
        x=zcords[:, 0],
        y=zcords[:, 1],
        z=zcords[:, 2],
        mode="lines",
        line=dict(color="green", width=5),
        name=name,
        legendgroup=name,
        hovertemplate="Z axis",
        showlegend=False,
    )
    fig.add_trace(x_axis)
    fig.add_trace(y_axis)
    fig.add_trace(z_axis)


def plot_world_coordinates(
    fig: go.Figure,
    xrange=(0, 10),
    yrange=(0, 10),
    zrange=(0, 10),
):
    # Add the x-axis
    fig.add_trace(
        go.Scatter3d(
            x=xrange, y=[0, 0], z=[0, 0], mode="lines", name="x-axis", showlegend=False
        )
    )

    # Add the y-axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0], y=yrange, z=[0, 0], mode="lines", name="y-axis", showlegend=False
        )
    )

    # Add the z-axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0], y=[0, 0], z=zrange, mode="lines", name="z-axis", showlegend=False
        )
    )
