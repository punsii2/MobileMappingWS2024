from pathlib import Path
from typing import Optional

import cv2 as cv
import streamlit as st


@st.cache_data()
def load_image(path: Path, scale: Optional[float], gray: bool = False):
    image = cv.imread(str(path))
    if scale and scale > 1:
        image = cv.resize(image, (0, 0), fx=scale, fy=scale)
    if gray:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image
