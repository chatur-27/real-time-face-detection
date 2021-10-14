# importing CV - computer vision package
import cv2
 
# getting trained face data from open cv library in github
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# to capture video from webcam
webcam = cv2.VideoCapture(0)

# loops through each frame of the video capture
while True:     
    
    # return two value boolean and each frame
    successful_frame_read,frame = webcam.read()
    
    # converting to grayscale for better recognition
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # comparing the grayscaled image with the trained data
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # looping through all the detected face and drawing rectangle over the original image
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    # displaying the frame in real time
    cv2.imshow("detecting",frame)
    
    # value pressed is stored in key
    key = cv2.waitKey(1)
    
    # to stop if 'q' or 'Q' is pressed
    if (key == 81 or key == 113):
        break
    
# to stop the webcam from broadcasting    
webcam.release()



