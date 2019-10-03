from imutils import face_utils
import dlib
import cv2
 
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0) #Jos tietokoneessa vain yksi kamera aseta arvoksi 0. Muuten voit valita kameran 1,2,3 jne.

line = []
 
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        nose = shape[27:36]
       
        # Draw on our image, all the finded cordinate points (x,y) 
        # for (x, y) in nose:
        #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        nose_point = nose[3]
        cv2.circle(image, (nose_point[0], nose_point[1]), 2, (0, 255, 0), -1)
        line.append(nose_point)

        print(shape)

    # Show the image
    # cv2.line(image, (x1, y1), (x2, y2), (0,255,0), lineThickness)
    for index in range(len(line) - 1):
        cv2.line(image, (line[index][0], line[index][1]), (line[index+1][0], line[index+1][1]), (0,0,255), 2)
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

