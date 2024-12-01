import cv2 as cv


def surf_match(image1, image2, max_ratio: float = 0.6):
    # Add the x-axis
    surf = cv.xfeatures2d.SURF_create(400, upright=True)
    keypoints1, destinations1 = surf.detectAndCompute(image1, None)
    keypoints2, destinations2 = surf.detectAndCompute(image2, None)

    # Match keypoints
    # BFMatcher with default params
    brute_force_matcher = cv.BFMatcher()
    matches = brute_force_matcher.knnMatch(destinations1, destinations2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < max_ratio * n.distance:
            good_matches.append([m])

    return good_matches, keypoints1, keypoints2


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    # cv.drawMatchesKnn expects list of lists as matches.
    return cv.drawMatchesKnn(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
