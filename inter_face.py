# libraries
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


# loading our images
kdot = face_recognition.load_image_file("imagos/kdot.jpg")
encode_kdot = face_recognition.face_encodings(kdot)[0]
maya = face_recognition.load_image_file("imagos/maya.jpg")
encode_maya = face_recognition.face_encodings(maya)[0]
messi = face_recognition.load_image_file("imagos/messi.jpg")
encode_messi = face_recognition.face_encodings(messi)[0]
gandhi = face_recognition.load_image_file("imagos/gandhi.jpg")
encode_gandhi = face_recognition.face_encodings(gandhi)[0]
mandela = face_recognition.load_image_file("imagos/mandela.jpg")
encode_mandela = face_recognition.face_encodings(mandela)[0]

known_faces = [encode_gandhi, encode_mandela, encode_kdot, encode_maya, encode_messi]
known_names = ['Lamar', 'Maya', 'Mandela', 'Gandhi', 'Messi']
our_students = known_names.copy()

fc_locations = [] # saves face locations from the webcam
fc_encodings = []
fc_names = []

# getting our current date and time
list_current_dt = (datetime.now()).strftime("%Y-%m-%d")

# creating attendance list instance
f = open(list_current_dt + ".csv", 'w+', newline='')
lnwriter = csv.writer(f)



# starting our webcam whose id is 0
webcam_capture = cv2.VideoCapture(0)


while True:
    signal, frame = webcam_capture.read() # reading our image input
    fremz = cv2.resize(frame, (0,0), fx=1, fy=1) #  decreasing the size
    rgb_fremz = fremz[:,:,::-1] # cv2 uses rgb
   
    # stores our new incoming image details
    fc_locations = face_recognition.face_locations(rgb_fremz)
    fc_encodings = face_recognition.face_encodings(rgb_fremz, fc_locations)

    for abc in fc_encodings:
        check_match = face_recognition.compare_faces(known_faces, abc)
        fc_distance = face_recognition.face_distance(known_faces, abc)
        best_match = np.argmin(fc_distance)

        if check_match[best_match]:
            name  = known_names[best_match]
        
        fc_names.append(name)
        if name in known_names:
            if name in our_students:
                our_students.remove(name)
                current_dt = (datetime.now()).strftime("%H:%M:%S")
                print(name)
                lnwriter.writerow([name, current_dt])

    cv2.imshow("Attendance System", fremz)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam_capture.release()
cv2.destroyAllWindows()
f.close()

