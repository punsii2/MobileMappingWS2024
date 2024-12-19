import os
from pathlib import Path

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import utils

EXERCISES = 6
TITLE = f"Project"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)
exercise = 5

tabs = st.tabs(list(map(lambda x: str(x), range(1, EXERCISES + 1))))

DATA_PATH = Path(os.getcwd() + "/../data/Proj/lab_10")
PATHS_LEFT = sorted((DATA_PATH / "mod/left/").glob("*.jpg"))
PATHS_RIGHT = sorted((DATA_PATH / "mod/right/").glob("*.jpg"))
PATHS_DISPARITY = sorted((DATA_PATH / "mod/disp/").glob("*.jpg"))
TRAJECTORY_PATH = (
    DATA_PATH / "tsukuba_ground_truth_poses/tsukuba_ground_truth_poses.txt"
)

exercise += 1
with tabs[exercise - 6]:
    st.write(f"## 7.2.{exercise}.1")

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
    matches_left = []
    for idx, disparity in enumerate(images_left[1:]):
        matches, keypoints1, keypoints2 = utils.surf_match(start_image_left, disparity)
        matches_left.append(matches)

        # cv.drawMatchesKnn expects list of lists as matches.
        image_matches = cv.drawMatchesKnn(
            start_image_left,
            keypoints1,
            disparity,
            keypoints2,
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
    matches_right = []
    for idx, disparity in enumerate(images_right[1:]):
        matches, keypoints1, keypoints2 = utils.surf_match(start_image_right, disparity)
        matches_right.append(matches)
        matched_images = utils.draw_matches(
            start_image_right, keypoints1, disparity, keypoints2, matches
        )
        columns = st.columns(2)
        ca2[idx + 1].image(matched_images)
        ca2[idx + 1].code(f"{len(matches)=}")

exercise += 1
with tabs[exercise - 6]:
    st.write(f"## 7.2.{exercise}.1")
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

        width, height, _ = image.shape
        colors = []
        for w in range(width):
            for h in range(height):
                colors.append(
                    f'rgb{np.array2string(image[w][h], separator=",").replace("[", "(").replace("]", ")")}'
                )

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

    camera_poses = utils.read_poses(TRAJECTORY_PATH)
    st.video(utils.image_to_video(images_left))
