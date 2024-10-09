from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly
import transforms3d as tr
from plotly import graph_objects as go


def plot_segment(begin, delta, k, color):
    """
    Function to plot segment in plotly
    :param begin - begin of vector
    :param delta - direction of vector
    :param k - length of vector
    :param color - color of vector
    """
    end = begin + k * delta
    return go.Scatter3d(
        x=[begin[0], end[0]],
        y=[begin[1], end[1]],
        z=[begin[2], end[2]],
        line=dict(color=color, width=2),
        mode="lines",
    )


def plot_poses(
    poses, k=0.3, text=None, marker_color=None, fontsize=None, only_points=False
):
    """
    Draw camera position with plotly
    :param poses - np.ndarray [N, 4, 4]
        poses of camera in system cam2world
    :param k - float or np.ndarray.
        length of the axis. default = 0.3
    :param text - List[str]
        title for every point
    :param marker_color
        color for every point
    :param fontsize
        fontsize for text
    :param only_points
        plot only points or camera directions too
    """
    # distance to point from (0, 0, 0)
    r = poses if only_points else poses[..., -1][:, [0, 1, 2]]

    # plot all origins
    fig = go.Figure(
        go.Scatter3d(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            marker=dict(size=3, color=marker_color),
            text=text,
            marker_symbol="diamond",
            mode="markers+text",
            textfont=dict(
                size=fontsize,
            ),
        ),
        layout=go.Layout(showlegend=False),
    )

    if isinstance(k, Iterable):
        kx = k[0]
        ky = k[1]
        kz = k[2]
    else:
        kx = ky = kz = k

    if not only_points:
        for pos in poses:
            xax = pos[:3, 0]
            yax = pos[:3, 1]
            zax = pos[:3, 2]
            point_begin = pos[:3, 3]
            fig.add_trace(plot_segment(point_begin, xax, kx, "red"))
            fig.add_trace(plot_segment(point_begin, yax, ky, "green"))
            fig.add_trace(plot_segment(point_begin, zax, kz, "blue"))
    return fig


def plot_cube(size, center):
    """
    plots cube with shape [size, size, size] as center
    and edges are parallel to coordinate axis
    """
    half_size = size / 2

    vertices = [
        [
            center[0] + half_size,
            center[1] + half_size,
            center[2] + half_size,
        ],  # Front top right
        [
            center[0] + half_size,
            center[1] - half_size,
            center[2] + half_size,
        ],  # Front bottom right
        [
            center[0] - half_size,
            center[1] - half_size,
            center[2] + half_size,
        ],  # Front bottom left
        [
            center[0] - half_size,
            center[1] + half_size,
            center[2] + half_size,
        ],  # Front top left
        [
            center[0] + half_size,
            center[1] + half_size,
            center[2] - half_size,
        ],  # Back top right
        [
            center[0] + half_size,
            center[1] - half_size,
            center[2] - half_size,
        ],  # Back bottom right
        [
            center[0] - half_size,
            center[1] - half_size,
            center[2] - half_size,
        ],  # Back bottom left
        [
            center[0] - half_size,
            center[1] + half_size,
            center[2] - half_size,
        ],  # Back top left
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Front face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Back face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Connecting edges
    ]

    layout = go.Layout(title="3D Cube", showlegend=False)
    fig = go.Figure(layout=layout)

    for e in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[vertices[e[0]][0], vertices[e[1]][0]],
                y=[vertices[e[0]][1], vertices[e[1]][1]],
                z=[vertices[e[0]][2], vertices[e[1]][2]],
                mode="lines",
                line=dict(width=2, color="blue"),
            )
        )

    return fig


def plot_path(r, marker_color=None, cut_color="black"):
    # plot all origins
    fig = go.Figure(
        go.Scatter3d(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            marker=dict(size=3, color=marker_color),
            marker_symbol="diamond",
            mode="markers",
        )
    )

    for i in range(1, len(r)):
        fig.add_trace(
            go.Scatter3d(
                x=[r[i - 1][0], r[i][0]],
                y=[r[i - 1][1], r[i][1]],
                z=[r[i - 1][2], r[i][2]],
                line=dict(
                    width=2,
                    color=cut_color,
                ),
                mode="lines",
            )
        )
    return fig


def combine_figs(*args):
    """
    to plot many figs at once
    """
    figs_data = [el for f in args for el in f.data]
    return figs_data


M = transformTranslation = tr.affines.compose([1, 2, 3], np.identity(3), np.ones(3))
plot_poses(np.ndarray((1, 4, 4))).show()
