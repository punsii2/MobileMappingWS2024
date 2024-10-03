import cv2

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.imshow("sobel", cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=11))
    # print(f"{frame.mean()=}")
    # print(f"{frame.max()=}")

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q") or key == 27:
        break

vid.release()
cv2.destroyAllWindows()
#
# cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)
#
# if vc.isOpened():  # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False
# while rval:
#     print(type(vc))
#     cv2.imshow("preview", frame)
#     rval, frame = vc.read()
#
# vc.release()
# cv2.destroyWindow("preview")
