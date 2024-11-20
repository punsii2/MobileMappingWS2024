import os
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st

EXERCISE = 6
TASKS = 2
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)
task = 0

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))

RED = [255, 0, 0]

DATA_PATH = Path(os.getcwd() + "/../data/LV_6/")
LEFT_PATH = DATA_PATH / "eiffel2-1.jpg"
RIGHT_PATH = DATA_PATH / "eiffel2-2.jpg"


task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    # read an image from disk
    image_left = cv.cvtColor(cv.imread(str(LEFT_PATH)), cv.COLOR_BGR2RGB)
    image_right = cv.cvtColor(cv.imread(str(RIGHT_PATH)), cv.COLOR_BGR2RGB)
    cl, cr = st.columns(2)
    cl.image(image_left)
    cr.image(image_right)

    def allCornersFromImage(image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        array = np.float32(gray_image)
        return cv.cornerHarris(array, 2, 3, 0.04)

    def draw_cross(image, x, y):
        top = max(x - 8, 0)
        bottom = min(x + 8, image.shape[0])
        left = max(y - 8, 0)
        right = min(y + 8, image.shape[1])
        cv.line(
            image,
            (left, top),
            (right, bottom),
            RED,
            2,
        )
        cv.line(
            image,
            (left, bottom),
            (right, top),
            RED,
            2,
        )
        # image_left_top_20[idx] = RED

    corners_left = allCornersFromImage(image_left)
    corners_right = allCornersFromImage(image_right)
    image_left_corners = image_left.copy()
    image_right_corners = image_right.copy()
    # Threshold for an optimal value, it may vary depending on the image.
    image_left_corners[corners_left > 0.01 * corners_left.max()] = RED
    image_right_corners[corners_right > 0.01 * corners_right.max()] = RED

    cl2, cr2 = st.columns(2)
    cl2.image(image_left_corners)
    cr2.image(image_right_corners)

    image_left_top_20 = image_left.copy()
    image_right_top_20 = image_right.copy()
    max_index_list_left = np.unravel_index(
        np.argpartition(corners_left.ravel(), -20)[-20:], corners_left.shape
    )
    max_indices_left = list(zip(max_index_list_left[0], max_index_list_left[1]))
    max_index_list_right = np.unravel_index(
        np.argpartition(corners_right.ravel(), -20)[-20:], corners_right.shape
    )
    max_indices_right = list(zip(max_index_list_right[0], max_index_list_right[1]))
    for idx in max_indices_left:
        draw_cross(image_left_top_20, idx[0], idx[1])
    for idx in max_indices_right:
        draw_cross(image_right_top_20, idx[0], idx[1])

    cl3, cr3 = st.columns(2)
    cl3.image(image_left_top_20)
    cl3.write(np.array(max_indices_left).transpose())
    cr3.image(image_right_top_20)
    cr3.write(np.array(max_indices_right).transpose())

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    surf = cv.xfeatures2d.SURF_create(400, upright=True)
    keypoints_left, destinations_left = surf.detectAndCompute(image_left, None)
    keypoints_right, destinations_right = surf.detectAndCompute(image_right, None)

    image_keypoints_left = cv.drawKeypoints(
        image_left, keypoints_left, None, (255, 0, 0), 4
    )
    image_keypoints_right = cv.drawKeypoints(
        image_right, keypoints_right, None, (255, 0, 0), 4
    )
    st.image(image_keypoints_left)
    st.image(image_keypoints_right)

    # Match keypoints
    # BFMatcher with default params
    brute_force_matcher = cv.BFMatcher()
    matches = brute_force_matcher.knnMatch(destinations_left, destinations_right, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    image_matches = cv.drawMatchesKnn(
        image_left,
        keypoints_left,
        image_right,
        keypoints_right,
        good_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    st.image(image_matches)
    st.code(f"{len(good_matches)=}")
