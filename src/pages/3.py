import os
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import streamlit as st

EXERCISE = 3
TASKS = 4
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))
task = 0


@st.cache_data()
def read_images(image_paths: List[Path]):
    images = [cv.imread(str(path), cv.COLOR_RGB2BGR) for path in image_paths]
    h, w, _ = images[0].shape
    return [cv.resize(image, (int(w / 2), int(h / 2))) for image in images]


@st.cache_data()
def find_pattern(image, pattern_size):
    processed_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return cv.findChessboardCornersSB(processed_image, pattern_size)


task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pattern_size = (9, 6)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[1], 0 : pattern_size[0]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    image_paths = list(Path(os.getcwd() + "/../data/LV_3/").glob("*"))
    images = read_images(image_paths)
    num_images = len(images)

    # st.header("Calibration Images:")
    # columns = st.columns(5)
    # i = 0
    # for img in images:
    #     columns[i].image(img)
    #     i = (i + 1) % 5
    # st.divider()

    results = []
    results.append(st.columns([2, 1, 1]))
    results[0][0].write("Calibration images")
    results[0][1].write("Rotation vectors")
    results[0][2].write("Translation vectors")
    index = 0
    successes = 0
    progress = st.progress(
        0,
        text=f"Finding checkerboard pattern...\nProcessing image {index} / {num_images}...",
    )
    for image in images:
        progress.progress(
            index / num_images, text=f"Processing image {index} / {num_images}..."
        )
        index += 1
        # Find the chess board corners
        return_value, corners = find_pattern(image, pattern_size)
        # corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        if return_value:
            successes += 1
            objpoints.append(objp)
            imgpoints.append(corners)

            results.append(st.columns([2, 1, 1]))
            results[successes][0].image(
                cv.drawChessboardCorners(image, pattern_size, corners, return_value)
            )
    progress.empty()
    st.write(f"Done! Found a pattern in {successes} out of {num_images} images.")
    st.divider()

    (
        return_value,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    ) = cv.calibrateCamera(
        objpoints, imgpoints, [image.shape[0], image.shape[1]], None, None
    )

    for idx, r in enumerate(rotation_vectors):
        results[idx + 1][1].write(r)

    for idx, t in enumerate(translation_vectors):
        results[idx + 1][2].write(t)

    matrix_columns = st.columns(2)
    with matrix_columns[0]:
        st.write("camera_matrix:")
        st.write(camera_matrix)
    with matrix_columns[1]:
        st.write("distortion_coefficients:")
        st.write(distortion_coefficients)

    # st.divider()
    # st.write("Undistortion:")
    # img = cv.imread("left12.jpg")
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
