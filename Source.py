import cv2

cascPath  = 'haarcascade_frontalface_default.xml' #Trained cascade classifier data by OpenCV
cap = cv2.VideoCapture(0) #Initializes the webcam

def faceDetect(img, img_rgb):
    
    #Loads the classifier
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    #Detect faces
    faces = faceCascade.detectMultiScale(
              img,
              scaleFactor=1.1,
              minNeighbors=5
        )
		
    #Draws the rectangle over the faces.
    for x,y,w,h in faces:
        cv2.rectangle(img_rgb, (x,y), (x + w, y + h), (0, 255, 0), 2)
        
    return img_rgb

def main():
    while(True):
		# Capture frame-by-frame from webcam source
        ret, frame = cap.read()
		
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  		 #Face detection operations
        gray = faceDetect(gray, frame)

        # Display the resulting frame
        cv2.imshow('Face Detect (Press Q to close the window.)', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
main()

