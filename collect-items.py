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

# (x, y)
collectables = [
    (50, 50),
    (100, 100),
    (200, 300)
    ]

COLLECTABLE_RADIUS = 20
 
while True:
    # Getting out image by webcam 
    _, image = cap.read()

    # flip the image
    image = cv2.flip(image, 1)

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
       
        # this is the center point in nose
        nose_point = nose[3]
        cv2.circle(image, (nose_point[0], nose_point[1]), 5, (0, 255, 0), -1)

        # check if touching collectable
        # a list for keeping track of the collectables NOT hit
        not_hit_cols = []
        for col in collectables:
            # calc min and max x-pos for the col
            x_min = col[0]
            x_max = col[0] + COLLECTABLE_RADIUS

            # calc min and max y-pos for the col
            y_min = col[1]
            y_max = col[1] + COLLECTABLE_RADIUS

            # check if nose_point x and y inside the ranges
            x_hit = False
            if nose_point[0] > x_min and nose_point[0] < x_max:
                x_hit = True

            y_hit = False
            if nose_point[1] > y_min and nose_point[1] < y_max:
                y_hit = True

            if x_hit and y_hit:
                print('HITTING collectable!')
            else:
                # keep this one still around
                not_hit_cols.append(col)

        # replace with those we want to keep
        collectables[:] = not_hit_cols

        # print(shape)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,25)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(image,'Use your nose to collect the blue circles!', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    # draw the collectables
    for item in collectables:
        cv2.circle(image, item, COLLECTABLE_RADIUS, (255,0,0), -1)

    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

