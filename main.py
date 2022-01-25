# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import sys
import os

module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
	sys.path.append(module_path)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the frame dimensions
	# build a blob from it
	# Height & Weight
	(h, w) = frame.shape[:2]
	# The blob
	blob = cv2.dnn.blobFromImage(frame,
								 1.0,
								 (224, 224),
								 (104.0, 177.0, 123.0) )

	# pass the blob to neural network to obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialization of:
	# our list of faces, their locations, list of predictions from our mask neralNet
	faces = []
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (= probability)
		# associated with the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by:
		# ensuring the confidence is > min confidence
		if confidence > 0.5:
			# compute the (x, y) :
			# coordinates of bounding-box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# ensure the bounding-boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			# extract the face ROI (Region of Interest)
				# (Region of Interest= detection technique of images)
			# convert it from BGR to RGB channel, ordering it, resize to 224x224 pixels
			# and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face & bounding-boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# make a predictions if **at least** one face was detected
	if len(faces) > 0:
		# for faster deduction;
		# we make batch predictions on all faces at the same time
		# rather than 1 by 1 predictions in the above *for* loop
		faces = np.array(faces, dtype= "float32")
		preds = maskNet.predict(faces, batch_size= 32)

	# return 2 tuples of the face locations and their locations=
	# locs = bounding box, preds = the accuracy
	return (locs, preds)

# Loading face detection model using faceNet
# 1- Load (deploy.prototxt) of the caffe model
# it contains the information regarding the input size
prototxtPath = r"deploy.prototxt"
# 2- Load the weights from caffe model
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our face mask detector model that we trained
maskNet = load_model("mask_detector_224x.model")

# initialize the video stream (load our camera)
print("[-- INFO --] Starting Video Stream...")
# src=0 (I have one camera),
# if you have 2 cameras & you want to use the 2nd use (src=1)
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
# using while to keep the camera open till we exit
while True:
	# Frame = from video (loading from our camera)
	# take the frame from the video stream and resize it
	# to have a max-width = 600pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	# detect faces in the frame and;
	# determine if they are wearing a Mask or Not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# Add class label & color used to draw the bounding-box & text
		# mask = GREEN rectangle, no-mask = RED rectangle
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (76, 153, 0) if label == "Mask" else (0, 0, 153)
		# include the probability in the label ( prediction percentage )
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label & bounding-box rectangle on the output frame
		cv2.putText(frame, label,
					(startX, startY - 10),
					cv2.FONT_HERSHEY_TRIPLEX, 0.54, (153,20,90), 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

	# show the output frame
	cv2.imshow("FRAME MASK DETECTOR", frame)
	key = cv2.waitKey(1) & 0xFF
	# breaking from while loop if "q" button is pressed é shut the camera
	if key == ord("q"):
		break

# cleanup by destroy windows and stop video streaming
cv2.destroyAllWindows()
vs.stop()

