import cv2

left_eye_point_x = 0
right_eye_point_x = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture(0)

while 1:
    # read frame and convert to grayscale
    retval, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # draw rectagle around detected faces
        cv2.rectangle(img, (x, y), (x + int(w/2), y + int(h/2)), (255, 0, 0), 2)

        # get regions of interest
        roi_gray = gray[y:y + int(h / 2), x:x + int(w/2)]
        roi_color = img[y:y + h, x:x + w]

        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # calculate center points of eyes
            if (ex + ey)/2 < (x + y)/2:
                left_eye_point_x = (ex + ey)/2
            elif (ex + ey)/2 > (x + y)/2:
                right_eye_point_x = (ex + ey)/2

            # draw line between eyes
            cv2.line(img,
                     (int(left_eye_point_x), int(right_eye_point_x)),
                     (int(left_eye_point_x), int(right_eye_point_x)),
                     (0, 0, 0),
                     10)

    # show frame
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
