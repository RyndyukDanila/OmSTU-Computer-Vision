import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("snapshot")
img_counter = 1

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("snapshot", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "images/{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter += 1

cam.release()
cv2.destroyAllWindows()