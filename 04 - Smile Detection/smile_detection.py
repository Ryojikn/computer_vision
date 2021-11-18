# Face recognition test
# Ryoji Kuwae Neto

import cv2

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


# Defining a detection function
def detect(gray, frame):
	smiles = smile_cascade.detectMultiScale(gray, 3, 20)
	for (x, y, w, h) in smiles:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2);
	return frame


video_capture = cv2.VideoCapture(0)
while True:
	_, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	canvas = detect(gray, frame)
	cv2.imshow('Video', canvas)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
	