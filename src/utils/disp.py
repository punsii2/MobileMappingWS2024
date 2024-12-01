import numpy as np


def disparity_to_points(disparity):
    disparity_vector = disparity.ravel()
    # in stereogemoetry disparity < 0 are not valid
    FOCAL_LENGTH = 0.08 * 2000
    CAMERA_DIST = 0.4 * 2000

    width, height = disparity.shape
    x = []
    y = []
    for w in range(width):
        for h in range(height):
            x.append(w - int(width / 2))
            y.append(h - int(height / 2))

    z = np.array(FOCAL_LENGTH * CAMERA_DIST / -disparity_vector)
    x = np.multiply(-z, np.array(x) / FOCAL_LENGTH)
    y = np.multiply(-z, np.array(y) / FOCAL_LENGTH)

    # filter_mask_result = (z > -2200) & (z < -400)
    # z = z[filter_mask_result]
    # x = x[filter_mask_result]
    # y = y[filter_mask_result]

    return [x, y, z]
