import os
from pathlib import Path
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plyfile import PlyData

EXERCISE = 1
TASKS = 5
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, TASKS + 1))))
task = 0

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    DATA_PATH = Path(os.getcwd() + "/../data/LV_1/")
    KARLSTR_PATH = DATA_PATH / "HM_Karlstr.jpg"

    # read an image from disk
    image = cv2.cvtColor(cv2.imread(str(KARLSTR_PATH)), cv2.COLOR_RGB2BGR)
    st.image(image)

    st.write(f"{image.shape=}" + " (rows, columns, channels)")

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(image_gray)

    historgram_gray = plt.figure()
    histogram = cv2.calcHist(image_gray, [0], None, [256], [0, 256])
    plt.hist(image_gray.ravel(), bins=256)
    st.write("Histogramm of grayscale image:")
    st.pyplot(historgram_gray)

    historgram_bgr = plt.figure()
    for i, col in enumerate(["b", "g", "r"]):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    st.write("Histogramm of colored image:")
    st.pyplot(historgram_bgr)

    st.markdown(
        """
    Interpretation:
    Histogramm is not uniformly distributed => bad image quality?
    """
    )

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    scale = 0.1
    image_small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    st.image(image_small)

    @st.cache_data
    def apply_black_mask(image, width):
        max_rows = image.shape[0]
        max_columns = image.shape[1]

        if len(image.shape) > 2:
            # top left
            image[0:width, 0:width, :] = 0
            # bottom left
            image[max_rows - width : max_rows, 0:width, :] = 0
            # top right
            image[0:width, max_columns - width : max_columns, :] = 0
            # bottom right
            image[max_rows - width :, max_columns - width : max_columns, :] = 0
            # middle
            image[
                int((max_rows - width) / 2) : int((max_rows + width) / 2),
                int((max_columns - width) / 2) : int((max_columns + width) / 2),
                :,
            ] = 0

        else:
            # top left
            image[0:width, 0:width] = 0
            # bottom left
            image[max_rows - width : max_rows, 0:width] = 0
            # top right
            image[0:width, max_columns - width : max_columns] = 0
            # bottom right
            image[max_rows - width :, max_columns - width : max_columns] = 0
            # middle
            image[
                int((max_rows - width) / 2) : int((max_rows + width) / 2),
                int((max_columns - width) / 2) : int((max_columns + width) / 2),
            ] = 0
        return image

    width = st.slider("Width of black mask:", 0, 400, 100)
    st.image(apply_black_mask(image, width))
    st.image(apply_black_mask(image_gray, width))


task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")
    st.button("Rerun animation")

    # images in grid
    images = []
    columns = st.columns(3)
    for i in range(1, 10):
        image = cv2.imread(str(DATA_PATH / f"000{i}.png"))
        images.append(image)

        columns[(i - 1) % 3].image(image)
        cv2.imwrite(str(DATA_PATH / f"output/000{i}.png"), image)
        sleep(1)

    # video
    height = images[0].shape[0]
    width = images[0].shape[1]
    video_path = DATA_PATH / "output/video.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"avc1"), 1, (width, height)
    )
    for image in images:
        writer.write(image)
    writer.release()

    with video_path.open("rb") as f:
        video = f.read()

    st.video(video)

task += 1
with tabs[task - 1]:
    st.write(f"## 7.1.{EXERCISE}.{task}")

    plydata = PlyData.read(DATA_PATH / "../LV_2/teapotOut.ply")
    points = plydata["vertex"]

    df = pd.DataFrame()
    df["x"] = points["x"]
    df["y"] = points["y"]
    df["z"] = points["z"]
    fig = px.scatter_3d(df, x="x", y="y", z="z")
    st.plotly_chart(fig)
