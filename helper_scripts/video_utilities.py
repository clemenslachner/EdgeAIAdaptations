import os
import numpy as np
import cv2
import argparse
import imutils

# Returns tuple consisting of
#   - fourcc code
#   - output file type
def convert_to_fourcc_code(codec_str):
  fourcc_code = ''
  output_file_extension = ''
  
  if not codec_str:
    return ('', '')
  
  if codec_str == "mjpg":
    # MJPG -> .avi
    fourcc_code = 'MJPG'
    output_file_extension = '.avi'
  elif codec_str == "wmv1":
    # WMV3 -> .wmv
    fourcc_code = 'WMV1'
    output_file_extension = '.wmv'
  elif codec_str == "mp4v":
    # MP4V -> .mp4
    fourcc_code = 'mp4v'
    output_file_extension = '.mp4'
  else:
    print("Warning: No supported codec was found for '" + args["codec"] + "'")

  # Convert fourcc char[] value to int code
  return (fourcc_code, output_file_extension)

# base project parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputVideo", type=str, help="path to source video")
ap.add_argument("-o", "--outputPath", type=str, help="Output path and filename of altered video, with no extension")
ap.add_argument("-g", "--gray", type=bool, default=False, help="Grayscales video")
ap.add_argument("-d", "--display", type=bool, default=False, help="Display video while converting it")
ap.add_argument("-l", "--length", type=int, help="Length of ouput video")
ap.add_argument("-w", "--width", type=int, help="Width of ouput video")
ap.add_argument("-c", "--codec", type=str, help="Codec of output video (mp4v, mjpg, wmv1)")
args = vars(ap.parse_args())

if not args["inputVideo"]:
  print("Error: No video source file was set!")
  quit()

# Create a VideoCapture object
source_filename, source_file_extension = os.path.splitext(args["inputVideo"])
cap = cv2.VideoCapture(args["inputVideo"])

# Check if video opened successfully
if (cap.isOpened() == False): 
  print("Warning: Unable to read video source.")
  quit()

# Default resolutions of the frame are obtained.
# The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

# Check which codec should be used
(fourcc_code, output_file_extension) = convert_to_fourcc_code(args["codec"])  

if not fourcc_code:    
  print("Using default codec of video source!\n")
  #default values, if no specific codec is set
  fourcc_code = frame_fourcc  
  output_file_extension = source_file_extension
else:
  fourcc_code = cv2.VideoWriter_fourcc(*fourcc_code)

if args["outputPath"]:
  output_file_path = args["outputPath"]
else:
  output_file_path = "./output_" + source_filename
  
output_file_path += output_file_extension

if args["length"]:
  frame_height = args["length"]

if args["width"]:
  frame_width = args["width"]

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(output_file_path, fourcc_code, frame_fps, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True:     
    # Check if grayscaling is needed
    if (args["gray"]):
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = np.dstack([frame, frame, frame])
    
    # Write the frame into the file 'output.avi'
    out.write(frame)

    if args["display"]:
      # Display the resulting frame    
      cv2.imshow('frame',frame)

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 

# sources:
# https://stackoverflow.com/questions/50037063/save-grayscale-video-in-opencv
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
# https://www.fourcc.org/codecs.php