import os
from pathlib import Path

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import utils
from utils.plot import init_figure, plot_camera

EXERCISES = 6
TITLE = "Project"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, EXERCISES + 1))))

DATA_PATH = Path(os.getcwd() + "/../data/Proj/lab40")
PATHS_LEFT = sorted((DATA_PATH / "mod/left/").glob("*.jpg"))
PATHS_RIGHT = sorted((DATA_PATH / "mod/right/").glob("*.jpg"))
PATHS_DISPARITY = sorted((DATA_PATH / "mod/disp/").glob("*.jpg"))
TRAJECTORY_PATH = (
    DATA_PATH / "tsukuba_ground_truth_poses/tsukuba_ground_truth_poses.txt"
)

exercise = 1
with tabs[exercise - 1]:
    st.write(f"## 7.2.{exercise}.1")
    st.write("### Extract and Match Features")

    scale = 0.3
    # read an image from disk
    images_left = [utils.load_image(p, scale=scale) for p in PATHS_LEFT]
    images_right = [utils.load_image(p, scale=scale) for p in PATHS_RIGHT]
    disparities = [utils.load_image(p, scale=scale, gray=True) for p in PATHS_DISPARITY]

    ca1, ca2 = [], []
    columns = st.columns(2)
    ca1.append(columns[0])
    ca2.append(columns[1])
    ca1[0].write("Matching left side images:")
    ca2[0].write("Matching right side images:")
    start_image_left = images_left[0]
    keypoint_matches_left_origin = []
    keypoint_matches_left_target = []

    for idx, image_left in enumerate(images_left[1:]):
        matches, keypoints_origin, keypoints_target = utils.surf_match(
            start_image_left, image_left
        )
        keypoint_matches_left_origin.append(
            np.array([keypoints_origin[match[0].queryIdx].pt for match in matches])
        )
        keypoint_matches_left_target.append(
            np.array([keypoints_target[match[0].trainIdx].pt for match in matches])
        )

        # cv.drawMatchesKnn expects list of lists as matches.
        image_matches = cv.drawMatchesKnn(
            start_image_left,
            keypoints_origin,
            image_left,
            keypoints_target,
            matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        columns = st.columns(2)
        ca1.append(columns[0])
        ca2.append(columns[1])

        ca1[idx + 1].image(image_matches)
        ca1[idx + 1].code(f"{len(matches)=}")

    start_image_right = images_right[0]
    keypoint_matches_right_origin = []
    keypoint_matches_right_target = []

    for idx, image_right in enumerate(images_right[1:]):
        matches, keypoints_origin, keypoints_target = utils.surf_match(
            start_image_right, image_right
        )
        keypoint_matches_right_origin.append(
            np.array([keypoints_origin[match[0].queryIdx].pt for match in matches])
        )
        keypoint_matches_right_target.append(
            np.array([keypoints_target[match[0].trainIdx].pt for match in matches])
        )

        # cv.drawMatchesKnn expects list of lists as matches.
        image_matches = cv.drawMatchesKnn(
            start_image_right,
            keypoints_origin,
            image_right,
            keypoints_target,
            matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        ca2[idx + 1].image(image_matches)
        ca2[idx + 1].code(f"{len(matches)=}")

exercise = 2
with tabs[exercise - 1]:
    st.write(f"## 7.2.{exercise}.1")
    st.write("### Dense Matching")

    cb1, cb2 = [], []
    CAMERA_INTRINSICS = [[615, 0, 320], [0, 615, 240], [0, 0, 1]]
    for idx, disparity in enumerate(disparities):
        image = images_left[idx]

        columns = st.columns(2)
        cb1.append(columns[0])
        cb2.append(columns[1])

        cb1[idx].image(image)
        cb2[idx].image(disparity)

        x, y, z = utils.disparity_to_points(disparity)
        colors = utils.image_to_rgb_strings(image)

        point_cloud = utils.init_figure(
            eye={"x": 0, "y": 0, "z": 1.0}, up={"x": 1, "y": 0, "z": 0}
        )
        points_reconstructed = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        if idx < 2:
            # In theory we still need to go back to world coordinates
            # points_reconstructed = points_reconstructed @ R1.T + T1
            point_cloud.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    name="test",
                    legendgroup="test",
                    marker=dict(
                        size=3,
                        color=colors,
                        line_width=0.0,
                    ),
                )
            )
            st.plotly_chart(point_cloud)
        break

    st.video(utils.image_to_video(images_left))

    camera_poses = utils.read_poses(TRAJECTORY_PATH)[: len(PATHS_LEFT)]

    points_sequence = []
    colors_sequence = []
    for idx, disparity in enumerate(disparities):
        colors_sequence.extend(utils.image_to_rgb_strings(images_left[idx]))

        x, y, z = utils.disparity_to_points(disparity)

        coordinates = (x, y, z, np.zeros(len(x)))
        normalized_points = np.concatenate(coordinates).reshape((-1, 4), order="F")
        world_points = normalized_points @ camera_poses[idx].T
        points_sequence.extend(world_points)

    point_cloud = utils.init_figure(
        eye={"x": 0, "y": 0, "z": 1.0}, up={"x": 1, "y": 0, "z": 0}
    )
    x = [p[0] for p in points_sequence]
    y = [p[1] for p in points_sequence]
    z = [p[2] for p in points_sequence]
    point_cloud.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="test",
            legendgroup="test",
            marker=dict(
                size=1,
                color=colors_sequence,
                line_width=0.0,
            ),
        )
    )
    st.plotly_chart(point_cloud)

exercise = 3
with tabs[exercise - 1]:
    st.write(f"## 7.2.{exercise}.1")
    st.write("### Ransac und E-Matrix berechnen")

    st.write(
        "Rang muss gleich 2 sein => Ein Eigenwert ist 0 (bzw. sehr, sehr klein) und 2 ungleich 0."
    )
    st.write("Linke Kamera:")

    # Find fundamental matrix and filter keypoints for the left images
    rotations_left = []
    translations_left = []
    for i in range(len(keypoint_matches_left_origin)):
        keypoints_origin = keypoint_matches_left_origin[i]
        keypoints_target = keypoint_matches_left_target[i]
        F, FMask = cv.findFundamentalMat(
            keypoints_origin,
            keypoints_target,
        )
        E, EMask = cv.findEssentialMat(
            keypoints_origin,
            keypoints_target,
        )

        valid_keypoints_origin = np.array([e[0] for e in keypoints_origin[EMask]])
        valid_keypoints_target = np.array([e[0] for e in keypoints_target[EMask]])
        st.code(f"{np.linalg.matrix_rank(F)=}")
        st.code(f"{np.linalg.eigvals(F)=}")
        st.code(f"{np.linalg.matrix_rank(E)=}")
        st.code(f"{np.linalg.eigvals(E)=}")

        _, R, t, _ = cv.recoverPose(
            E,
            valid_keypoints_origin,
            valid_keypoints_target,
            np.array(CAMERA_INTRINSICS),
        )
        rotations_left.append(R)
        translations_left.append([tr[0] for tr in t])
        columns = st.columns(2)
        columns[0].write("F")
        columns[0].write(F)
        columns[1].write("E")
        columns[1].write(E)
        st.divider()

    st.write("Rechte Kamera:")

    # Find fundamental matrix and filter keypoints for the left images
    rotations_right = []
    translations_right = []
    for i in range(len(keypoint_matches_right_origin)):
        keypoints_origin = keypoint_matches_right_origin[i]
        keypoints_target = keypoint_matches_right_target[i]
        F, FMask = cv.findFundamentalMat(
            keypoints_origin,
            keypoints_target,
        )
        E, EMask = cv.findEssentialMat(
            keypoints_origin,
            keypoints_target,
        )

        valid_keypoints_origin = np.array([e[0] for e in keypoints_origin[EMask]])
        valid_keypoints_target = np.array([e[0] for e in keypoints_target[EMask]])
        _, R, t, _ = cv.recoverPose(
            E,
            valid_keypoints_origin,
            valid_keypoints_target,
            np.array(CAMERA_INTRINSICS),
        )
        rotations_right.append(R)
        translations_right.append([tr[0] for tr in t])
        columns = st.columns(2)
        columns[0].write("F")
        columns[0].write(F)
        columns[1].write("E")
        columns[1].write(E)
        st.divider()

exercise = 4
with tabs[exercise - 1]:
    st.write(f"## 7.2.{exercise}.1")
    st.write("### Posen berechnen ")

    fig = init_figure()

    for i in range(len(rotations_left)):
        R = rotations_left[i]
        t = translations_left[i]
        W = np.array(
            [
                [R[0, 0], R[0, 1], R[0, 2], t[0]],
                [R[1, 0], R[1, 1], R[1, 2], t[1]],
                [R[2, 0], R[2, 1], R[2, 2], t[2]],
                [0, 0, 0, 1],
            ]
        )
        plot_camera(fig, W, name="Left Camera", color="blue")

    for i in range(len(rotations_right)):
        R = rotations_right[i]
        t = translations_right[i]
        W = np.array(
            [
                [R[0, 0], R[0, 1], R[0, 2], t[0]],
                [R[1, 0], R[1, 1], R[1, 2], t[1]],
                [R[2, 0], R[2, 1], R[2, 2], t[2]],
                [0, 0, 0, 1],
            ]
        )
        plot_camera(fig, W, name="Right Camera", color="red")

    st.plotly_chart(fig)
