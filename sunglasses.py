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
       
       # eye locations found from
       # https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py
        eyes = shape[36:48]
        for (x, y) in eyes:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        print(shape)

    # Show the image
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

