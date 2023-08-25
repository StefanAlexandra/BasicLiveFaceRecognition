import threading
import cv2
from deepface import DeepFace

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

counter = 0
face_match = False
reference_image_path = input("Enter the path of the reference image: ")
reference_image = cv2.imread(reference_image_path)


def check_face(frame, reference_image):
    global face_match
    try:
        if DeepFace.verify(frame, reference_image)['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    ret, frame = capture.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(), reference_image.copy())).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, 'Match!', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No match!', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('video', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()



