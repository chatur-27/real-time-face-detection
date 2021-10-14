# importing CV - computer vision package
import cv2
 
# getting trained face data from open cv library in github
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# to read an image
img = cv2.imread("dinesh.jpg")

# to display an image 
cv2.imshow("check",img)
# similar to getch() to see the output
cv2.waitKey()

# converting to grayscale for better recognition
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#displaying grayscaled image
cv2.imshow("grayscaled image",grayscaled_img)
cv2.waitKey()

# comparing the grayscaled image with the trained data
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# looping through all the detected face and drawing rectangle over the original image
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# displaying the image with rectangle drawn over the face
cv2.imshow("detecting",img)
cv2.waitKey()

