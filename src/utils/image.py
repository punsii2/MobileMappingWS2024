import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np
import streamlit as st


@st.cache_data
def load_image(path: Path, scale: Optional[float], gray: bool = False) -> np.ndarray:
    image = cv.imread(str(path))
    if scale and scale != 1.0:
        image = cv.resize(image, (0, 0), fx=scale, fy=scale)
    if gray:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


@st.cache_data(persist=True)
def image_to_video(images):
    height = images[0].shape[0]
    width = images[0].shape[1]
    tmppath = Path(f"/tmp/{uuid.uuid4()}.mp4")
    tmppath.touch()

    writer = cv.VideoWriter(
        str(tmppath), cv.VideoWriter_fourcc(*"avc1"), 5, (width, height)
    )
    for image in images:
        writer.write(image)
    writer.release()

    with tmppath.open("rb") as f:
        video = f.read()
    return video


@st.cache_data(persist=True)
def read_poses(path: Path) -> List[np.ndarray]:
    poses = []
    with path.open() as f:
        for row in f:
            poses.append(
                np.array(
                    [float(x) for x in str(row).split(" ")] + [0.0, 0.0, 0.0, 1.0]
                ).reshape([4, 4])
            )
    return poses
