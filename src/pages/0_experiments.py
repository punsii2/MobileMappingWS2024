import glob
import os
from math import atan2, cos, pi, sin
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import streamlit as st
import transforms3d as tr
from plyfile import PlyData
from streamlit_back_camera_input import back_camera_input

EXERCISE = 0
TASKS = 1
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))
task = 0

task += 1


#
# picture = st.camera_input("Take a picture")
#
# if picture:
#     image = cv2.cvtColor(cv2.imread(str(KARLSTR_PATH)), cv2.COLOR_RGB2BGR)
#     st.image(picture)
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    st.write(f"## (Experiments)")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    image_buffer = back_camera_input()
    if image_buffer:
        st.image(image_buffer)

        bytes_data = image_buffer.getvalue()
        img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        st.image(img)
