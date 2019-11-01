from imutils import face_utils
import dlib
import cv2
 
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

line = []
 
while True:
    _, image = cap.read()

    image = cv2.flip(image, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    rects = detector(gray, 0)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        nose = shape[27:36]
        nose_point = nose[3]
        cv2.circle(image, (nose_point[0], nose_point[1]), 2, (0, 255, 0), -1)
        line.append(nose_point)

    for index in range(len(line) - 1):
        cv2.line(image, (line[index][0], line[index][1]), (line[index+1][0], line[index+1][1]), (0,0,255), 2)
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

