import os
import sys
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import streamlit as st

EXERCISE = 3
TASKS = 3
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)
task = 0


@st.cache_data()
def read_images(image_paths: List[Path]):
    return [cv.imread(str(path), cv.COLOR_RGB2BGR) for path in image_paths]


@st.cache_data()
def find_pattern(image, pattern_size):
    processed_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return cv.findChessboardCornersSB(processed_image, pattern_size)


def load_images(upload=False):
    images = []
    if upload:
        uploaded_files = st.file_uploader(
            "Choose image files", type=["png", "jpg"], accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                image_bytes = np.asarray(bytearray(f.read()), dtype="uint8")
                images.append(cv.imdecode(image_bytes, cv.COLOR_RGB2GRAY))
    else:
        image_paths = list(Path(os.getcwd() + "/../data/LV_3/").glob("*.png"))
        image_paths += list(Path(os.getcwd() + "/../data/LV_3/").glob("*.jpg"))
        images = read_images(image_paths)
    if images:
        h, w, _ = images[0].shape
        return [cv.resize(image, (int(w / 2), int(h / 2))) for image in images]
    return


upload = st.sidebar.checkbox("Upload your own images.")
images = load_images(upload)
if not images:
    sys.exit()
num_images = len(images)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    SQUARE_SIZE_MM = 40
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pattern_size = (9, 6)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[1], 0 : pattern_size[0]].T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

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
    if not successes:
        st.text(
            """
            None of the images can be used for calibration :(
            Please use different images.
        """
        )
        sys.exit()
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

task += 1
with tabs[task - 1]:
    st.write(
        "At this point I do not have access to Matlab or the required images that are part of the Vision Toolbox."
    )

task += 1
with tabs[task - 1]:
    st.header("Undistortion")
    h, w = images[0].shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, (w, h), 1, (w, h)
    )

    distortions = [st.columns(2)]
    distortions[0][0].write("Original Images")
    distortions[0][1].write("Undistorted Images")
    for idx, image in enumerate(images):
        # undistort
        undistorted = cv.undistort(
            image, camera_matrix, distortion_coefficients, None, newcameramtx
        )

        # # crop the image
        # x, y, w, h = roi
        # undistorted = undistorted[y : y + h, x : x + w]
        distortions.append(st.columns(2))
        distortions[idx + 1][0].image(image)
        distortions[idx + 1][1].image(undistorted)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i],
            rotation_vectors[i],
            translation_vectors[i],
            camera_matrix,
            distortion_coefficients,
        )
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    st.write(f"Total reprojection error: {format(mean_error / len(objpoints))}")

    st.text(
        """
        The original (distorted) and undistorted images differ more than i anticipated.
        The distortion parameters are all very close to 0, which is to be expected
        as the phone camera that was used to take the pictures should already be calibrated.
        However, the undistorted images still show quite a large black arc along the top and bottom.
    """
    )
