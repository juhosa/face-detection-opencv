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

sg = cv2.imread('sunglasses_PNG150_small.png', cv2.IMREAD_UNCHANGED)
# sg = cv2.imread('sunglasses_PNG150_small.png')

if sg is None:
    print('Error loading sunglasses')
    exit(1)

sg_rows, sg_cols, sg_channels = sg.shape


 
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
       # ("right_eye", (36, 42)),
       # ("left_eye", (42, 48)),
       # ("nose", (27, 36)),
        eyes = shape[36:48]
        for (x, y) in eyes:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # left (on screen) eye x = eyes[0][0]
        # left (on screen) eye y = eyes[0][1]
        # right (on screen) eye x = eyes[1][0]
        # right (on screen) eye y = eyes[1][1]
        print(eyes[0], eyes[-1])
        # cv2.circle(image, (eyes[0][0], eyes[0][1]), 2, (0, 255, 0), -1)
        
        # x_offset=y_offset=50
        x_offset = eyes[0][0] - 90
        y_offset = eyes[0][1] - 60

        y1, y2 = y_offset, y_offset + sg.shape[0]
        x1, x2 = x_offset, x_offset + sg.shape[1]

        alpha_sg = sg[:, :, 3] / 255.0
        alpha_image = 1.0 - alpha_sg

        for c in range(3):
            image[y1:y2, x1:x2, c] = (alpha_sg * sg[:, :, c] + 
            alpha_image * image[y1:y2, x1:x2, c])


        # print(shape)



    # Show the image
    cv2.imshow("Output", image)
    # cv2.imshow('Output', sg)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

