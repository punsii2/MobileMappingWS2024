import tempfile
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import streamlit as st


@st.cache_data
def load_image(path: Path, scale: Optional[float], gray: bool = False):
    image = cv.imread(str(path))
    if scale and scale != 1.0:
        image = cv.resize(image, (0, 0), fx=scale, fy=scale)
    if gray:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


# def image_to_video(images: List[cv.Image]):
@st.cache_data
def image_to_video(images):
    height = images[0].shape[0]
    width = images[0].shape[1]

    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    tmpfile.close()
    writer = cv.VideoWriter(
        str(tmpfile), cv.VideoWriter_fourcc(*"avc1"), 1, (width, height)
    )
    for image in images:
        writer.write(image)
    writer.release()

    with tmpfile.open("rb") as f:
        video = f.read()
    return video
