import cv2
face_cap=cv2.CascadeClassifier("C:/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
vid_cap = cv2.VideoCapture(0)
while True :
    ret , vid_data = vid_cap.read()
    col = cv2.cvtColor(vid_data,cv2.COLOR_BGR2GRAY)
    faces =face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(vid_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",vid_data)
    if cv2.waitKey(10) == ord("a"):
        break
vid_cap.release()


