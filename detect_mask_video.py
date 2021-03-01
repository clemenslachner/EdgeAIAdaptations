import os
# prevent python warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# prevent tensorflow hardware messages
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# prevent tensorflow logging
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import numpy as np
import argparse
import time
import cv2
import json
import time
import imutils

from models.tf.utils import label_map_util


complete_acc = 0.0
analysedFrameCount = 0


# preprocess frame if needed and calls analyse on a frame for tf and normal mode
def run_analyse_on_frame(frame, frameCount):
	timingDict = {}
	if(args["width"]):
		start_time_resizing = time.time()
		frame = imutils.resize(frame, width=args["width"])
		elapsed_time_resizing = time.time() - start_time_resizing
		timingDict['resizing'] = elapsed_time_resizing
	if(args["gray"]):
		# to grayscale and format since faceNet needs 3 channel image (setting all channels to graychannel) 
		start_time_grayscaling = time.time()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = np.dstack([frame, frame, frame])
		elapsed_time_grayscaling = time.time() - start_time_grayscaling
		timingDict['grayscaling'] = elapsed_time_grayscaling
	# detect faces in the frame and determine if they are wearing a face mask or not
	if(args["useTf"]):
		start_time_tf_inferencing = time.time()
		(locs, preds) = detect_and_predict_mask_tf(frame, maskNet)
		elapsed_time_inferencing = time.time() - start_time_tf_inferencing
		timingDict['inferencing-tf'] = elapsed_time_inferencing
	else:
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	frameAcc = 0
	maskCnt = 0
	noMaskCnt = 0
	anonymization = args["anonymization"]

	if((args["display"] and args["boxes"])):
		# loop over the detected face locations and their corresponding locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			acc = max(mask, withoutMask)
			frameAcc += acc

			# determine the class label and color we'll use to draw the bounding box and text
			if mask > withoutMask: 
				maskCnt += 1
				label = "Mask"
				color = (0, 255, 0)
			else: 
				noMaskCnt += 1
				label = "No Mask"
				color = (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}".format(label, acc)

			center = ((startX + endX) / 2, (startY + endY) / 2)
			radius = (startX + endX) / 2

			# display the label and bounding box rectangle on the output frame
			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			#cv2.rectangle(frame, (startX, startY), (endX, endY), color, -1)

			# blur faces in frame
			face = frame[startY:endY, startX:endX]
			#face = anonymize_face_simple(face, factor=3.0)
			face = anonymize_face_pixelate(face, blocks=10)
			frame[startY:endY, startX:endX] = face

			#cv2.circle(frame, center, 30, color, 2, 8, 0)

	elif(anonymization > 0):
        # loop over the detected face locations and their corresponding locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			if mask > withoutMask: maskCnt += 1
			else: noMaskCnt += 1
			frameAcc += max(mask, withoutMask)

			# anonymize faces in frame
			start_time_anoymization = time.time()
			face = frame[startY:endY, startX:endX]
			if(anonymization == 1):
				color = (0, 0, 255)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, -1)
				elapsed_time_anonymization = time.time() - start_time_anoymization
			elif(anonymization == 2):
				face = anonymize_face_pixelate(face, blocks=10)
				frame[startY:endY, startX:endX] = face
				elapsed_time_anonymization = time.time() - start_time_anoymization
			elif(anonymization == 3):
				face = anonymize_face_simple(face, factor=3.0)
				frame[startY:endY, startX:endX] = face
				elapsed_time_anonymization = time.time() - start_time_anoymization
			else:
				print('unknown anonymization strategie. no anonymization will be done')
				elapsed_time_anonymization = time.time() - start_time_anoymization
			
			timingDict['anonymization'] = elapsed_time_anonymization
	else:
		for pred in preds:
			(mask, withoutMask) = pred
			if mask > withoutMask: maskCnt += 1
			else: noMaskCnt += 1
			frameAcc += max(mask, withoutMask)

	return maskCnt, noMaskCnt, frameAcc * 100, timingDict

def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image

def anonymize_face_simple(image, factor=3.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size
	return cv2.GaussianBlur(image, (kW, kH), 0)

# preprocess frame if needed and calls analyse on a frame for tflite mode. Face and mask detection is done in one step
def run_analyse_on_frame_tflite(frame, frameCount):

	imH, imW, _ = frame.shape

	# we need to preprocess the image since the tflite model needs these params strictly
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb, (300, 300))
	input_data = np.expand_dims(frame_resized, axis=0)
	input_data = (np.float32(input_data) - 127.5) / 127.5	# intput_mean / input_std

	# actual detection of faces and Masks
	interpreter.set_tensor(input_details[0]['index'], input_data)
	start_time = time.time()
	interpreter.invoke()
	elapsed_time = time.time() - start_time
	print('Processing time: {}'.format(elapsed_time))

	boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
	classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
	scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

	# cut and count faces and masks
	noMaskCnt = 0
	maskCnt = 0
	frameAcc = 0
	for i in range(len(scores)):
		if ((scores[i] > args["confidence"]) and (scores[i] <= 1.0)):
			ymin = int(max(1,(boxes[i][0] * imH)))
			xmin = int(max(1,(boxes[i][1] * imW)))
			ymax = int(min(imH,(boxes[i][2] * imH)))
			xmax = int(min(imW,(boxes[i][3] * imW)))

			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

			if(classes[i] == 1):
				maskCnt += 1
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
			else:
				noMaskCnt += 1
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
			frameAcc += scores[i]


	return maskCnt, noMaskCnt, frameAcc * 100

# analyse a frame for faces with tf model and calls masknet detection
def detect_and_predict_mask_tf(frame, maskNet):

	image_np_expanded = np.expand_dims(frame, axis=0)
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	start_time = time.time()
	(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
	elapsed_time = time.time() - start_time
	if(args["verbose"]):
		print('Processing time FACE: {}'.format(elapsed_time))

	faces = []
	locs = []
	im_height, im_width, _ = frame.shape
	for i in range(int(num_detections[0])):
		if(scores[0][i] >= args["confidence"]):
			(xmin, xmax, ymin, ymax) = (int(boxes[0, i, 1] * im_width), int(boxes[0, i, 3] * im_width), int(boxes[0, i, 0] * im_height), int(boxes[0, i, 2] * im_height))
			faces.append(cut_face(frame, xmin, xmax, ymin, ymax))
			locs.append((xmin, ymin, xmax, ymax))
		else:
			break # resulting scores are sorted

	return predict_mask(faces, locs)

# analyse a frame for faces with normal model and calls masknet detection
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame
	(h, w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, mean=(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)

	start_time = time.time()
	detections = faceNet.forward()
	elapsed_time = time.time() - start_time
	if(args["verbose"]):
		print('Processing time FACE: {}'.format(elapsed_time))

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	for i in range(0, detections.shape[2]):
		# extract the confidence associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute coordinates of the bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the frame dimensions
			(startX, startY) = (min(w - 1, startX), min(h - 1, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			if (startX == endX or startY == endY):
				continue

			# extract the face and preprocess for masknet
			faces.append(cut_face(frame, startX, endX, startY,endY))
			locs.append((startX, startY, endX, endY))

	return predict_mask(faces, locs)

# analyse frame for masks
def predict_mask(faces, locs):
	preds = []
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")

		if not args["onlyFaces"]:
			start_time = time.time()
			preds = maskNet.predict(faces, batch_size=32)
			elapsed_time = time.time() - start_time
			if(args["verbose"]):
				print('Processing time MASK: {}'.format(elapsed_time))
		else:
			# [[1.0, 0.0]] tuple pair of lables (now all faces would be defined as without mask)
            # print("processing faces only!")
			preds = [[0.0, 1.0]] * faces.shape[0]
			
	return (locs, preds)

# cut out face on given positions from frame 
def cut_face(frame, xmin, xmax, ymin, ymax):
	face = frame[ymin:ymax, xmin:xmax]
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	
	return preprocess_input(face)

# add all infos of a frame to the result dictionary 
def add_frameinfo_to_result(frameId, masks, no_masks, accSum, timingDict):
	nrOfDetections = masks+no_masks

	if accSum != 0: avg_acc = accSum / nrOfDetections
	else: avg_acc = 0.0

	global complete_acc
	complete_acc = complete_acc + avg_acc

	if(args["verbose"]):
		print("[INFO] Frame {:d} | Faces found {:d} | With mask: {:d} | Without mask: {:d} | frame_acc: {:.2f}%".format(frameId, nrOfDetections, masks, no_masks, avg_acc) + " | " + str(timingDict))

	frameInfo[frameId] = {"faces": nrOfDetections,"masks": masks,"no_masks": no_masks,"accuracy": avg_acc, "timings": timingDict}

# handle the resulting dictionary of the analyse
def handle_result(elapsedTime, frameCount):
	if(args["verbose"]):
		print("[RESULT] elasped time: {:.2f}".format(elapsedTime))
		print("[RESULT] approx. FPS: {:.2f}".format(frameCount / elapsedTime))

	global analysedFrameCount
	info = {
		"runtime": elapsedTime,
		"fps": frameCount / elapsedTime,
		"complete_acc": complete_acc / analysedFrameCount,
		"arguments": [ {'name': n, 'value': args[n]} for n in args],
		"frames": [ {'frameId': n, 'frameMetrics': frameInfo[n]} for n in frameInfo]
	}

	infoAsJson = json.dumps(info, indent=4)

	if(args["output"]):
		with open(args["output"], 'w') as fileWriter:
			fileWriter.write(infoAsJson)
	
	if(args["print"]):
		print (infoAsJson)

# needed for switching between true and false values for arguments in the test framwork
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# base project parameters
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,default="models/face_detector",help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,default="models/mask_detector.model",help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
# own parameters
ap.add_argument("-s", "--camSource", type=str, default="/dev/video0", help="Camera id source in the system")
ap.add_argument("-i", "--inputVideo", type=str, help="path to video if needed")
ap.add_argument("-w", "--width", type=int, help="resize the input video frames")
ap.add_argument("-v", "--verbose", type=bool, default=False, help="print frame information during run")
ap.add_argument("-g", "--gray", type=boolean_string, help="preprocess each frame by grayscaling")
ap.add_argument("-b", "--boxes", type=bool, default=False, help="show detection boxes on the output video")
ap.add_argument("-d", "--display", type=bool, default=False, help="display the output video on screen")
ap.add_argument("-n", "--checkFrame", type=int, default=1, help="each n-th frame will be analyzed")
ap.add_argument("-o", "--output", type=str, help="ouput of the analyse result in json. Needs the path with filename")
ap.add_argument("-p", "--print", type=str, help="print ouput of the analyse result to the terminal")
ap.add_argument("-gpu", "--gpu", type=str, default="", help="device ids of gpu's to be used in the formet id,id,...")
ap.add_argument("-useTf", "--useTf", type=bool, default=False, help="use TF model for face detection instead of caffeemodel")
ap.add_argument("-useTflite", "--useTflite", type=bool, default=False, help="use TF-lite model for face detection instead of caffeemodel")
ap.add_argument("-onlyFaces", "--onlyFaces", type=bool, default=False, help="ignore masks on TF-model and only search for faces")
ap.add_argument("-a", "--anonymization", type=int, default=0, help="0...no anoymization, 1...simpleRect, 2...pixelate, 3...gaussianBlur")
args = vars(ap.parse_args())

# GPU will be prevered for detection. If no device is visible the cpu will be used (tensorflow-gpu is needed)
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]

# load our serialized face detector model from disk if needed
if (args["useTf"]):
	label_map = label_map_util.load_labelmap("models/tf/protos/face_label_map.pbtxt")
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	detection_graph = tf.Graph()
	
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()

		with tf.io.gfile.GFile("models/tf/frozen_inference_graph_face.pb", 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.compat.v1.Session(graph=detection_graph, config=config)
elif (args["useTflite"]):
	from tensorflow.lite.python.interpreter import Interpreter
	interpreter = Interpreter(model_path="models/tflite/final_model.tflite")
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	color_box = [(0,255,0), (0,0,255), (255,255,0)]
	with open("models/tflite/label_map.pbtxt", 'r') as f:
		labels = [line.strip() for line in f.readlines()]
	if labels[0] == '???':
		del(labels[0])

	floating_model = (input_details[0]['dtype'] == np.float32)
else:
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
if(args["inputVideo"]):
 	vs = FileVideoStream(args["inputVideo"]).start()
else:
	vs = VideoStream(src=args["camSource"]).start()
	time.sleep(2.0)

# loop over the frames from the video stream and create Dict
frameInfo = {} # filled by add_frameinfo_to_result(...)
startTime = time.time()
try:
	frameCount = 1
	analysedFrameCount = 1
	while True:

		frame = vs.read()
		if frame is None:
			break

		if(frameCount % args["checkFrame"] == 0):
			if(args["useTflite"]):
				maskCnt, noMaskCnt, frameAcc = run_analyse_on_frame_tflite(frame, frameCount)
			else:
				start_time_frame_analyze = time.time()
				maskCnt, noMaskCnt, frameAcc, timingDict = run_analyse_on_frame(frame, frameCount)
				elapsedTime_frame_analyze = time.time() - start_time_frame_analyze
				# add timing for one complete frame analyzation step
				timingDict['completeStep'] = elapsedTime_frame_analyze

			add_frameinfo_to_result(frameCount, maskCnt, noMaskCnt, frameAcc, timingDict)
			analysedFrameCount += 1
			
			# skip first frame for time tracking to prevent tracking initial overhead from tf
			if(analysedFrameCount == 2):
				startTime = time.time()

		if(args["display"]):
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

		frameCount += 1
except KeyboardInterrupt:
	pass 

elapsedTime = time.time() - startTime

cv2.destroyAllWindows()
vs.stop()

handle_result(elapsedTime, frameCount)
