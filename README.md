# Face Blurring VAP
Measure Performance of an AI-Assisted Video Analytics Pipeline (VAP) for Edge Computing.
The main goal is to identfy adaptations that potentially affect data protection quality.
The implementation is capable of detecting faces in video data (and masks) and applying different anonymization techniques to each frame.
# Install
- Install python 3 on the Raspberry (tested with python3.7, python3.6)
- Install openCV dependencies by using the provided script
>sh install/install.sh
- Install Virtualenv and create a virtual python3 environment and start it
>virtualenv -p python3 envname
>source envname/bin/activate
- Install the requirements into the virtual environment
>pip3 install -r install/requirements.txt

<br/><br/>

# detect_mask_video.py

### Livestream
> python3 detect_mask_video.py -s "cameraSource" (+ additional arguments)

### Video
> python3 detect_mask_video.py -i "videoFile" (+ additional arguments)

### Arguments
- Get list of input arguments by using 
> python3 detect_mask_video_custom.py -h

| Argument  | Name | Default | Description | 
|---|---|---|---|
| -s  | camSource | "/dev/video=0" | Source of the Camera in  the system (id) |
| -i  | inputVideo | --- | Path to video which should be analysed |
| -w | width | --- | Value to resize the input video during analyse. Aspect ratio remains |
| -v | verbose | False | Additional output during analyse |
| -g | gray | --- | Grascale frames before analysing |
| -b | boxed | False | Show boxes on displayed video (needs -d) |
| -d | display | False | Show analysed video |
| -n | checkFrame | 1 | Define which frames should be analyse. Each Nth frame. |
| -o | output | --- | Output path to save the resulting json |
| -p | print | --- | Print the resulting json |
| -gpu | gpu | "" | Id of the Gpu in the system which sould be used for analysing |
| -useTf | useTf | False | Use Tensorflow Model for Face-Detection instead of Caffeemodel |
| -useTflite | useTflite | False | Use TFlite Model for Face and Mask Detection instead of Caffeemodel |
| -onlyFaces | onlyFaces | False | Will skip the mask detection procedure |

<br/><br/>

# video_utilities.py

## Convert video
> python3 video_utilities.py -i "<input_file_path>" (+ additional arguments)

## Arguments
- Get list of input arguments by using 
> python3 video_utilities.py -h

| Argument  | Name | Default | Description | 
|---|---|---|---|
| -i  | inputVideo | --- | Path to source video |
| -o  | outputPath | --- | Output path and filename of altered video, with no extension. Default is "output_" + original_filename + "." + original_file_extension |
| -g | gray | False | Grascale frames before analysing |
| -d | display | False | Display video while converting it |
| -l | length | --- | Length of ouput video |
| -w | width | --- | Width of ouput video |
| -c | codec | --- | Codec of output video. Possible codecs: mp4v, mjpg, wmv1. When null, default codec von input file. |

<br/><br/>

# test_mask_detector.py

## Test the `detect_mask_video.py` script
> python3 test_mask_detector.py -n 30 -n 15 --videoFolder `"a folder with videos to check"`  --outputFolder `"the results folder location"` (+ additional arguments)

## Generate baselines
> python3 test_mask_detector.py --videoFolder `"a folder with videos to check"` --generateBaseline  --outputFolder `"the results folder location"` (+ additional arguments)

## Arguments
- Get list of input arguments by using 
> python3 test_mask_detector.py -h

| Argument  | Name | Default | Description | 
|---|---|---|---|
| -w  | width | --- | Multiple widths to be checked in the test run (Crossproduct) |
| -g | gray | False | Grascale frames before analysing (Crossproduct) |
| -n | checkFrame | 1 | Define which frames should be analyse. Each Nth frame. (Crossproduct) |
| -i | videoFolder | --- | the path to the videos folder |
| -v | videos | --- | specify multiple videos to check instead of using the videoFolder |
| -b | generateBaseline | False | generate the baselines (ignores all crossproducts and runs the baseline config for every video) |
| -o | outputFolder | results | the output folder for the `.json` files (if the folder doesn't exists it gets created) |
| -verbose | verbose | False | Additional output during testing |
| -gpu | gpu | "" | Id of the Gpu in the system which sould be used for analysing |
| -useTf | useTf | False | Use Tensorflow Model for Face-Detection instead of Caffeemodel |
| -useTflite | useTflite | False | Use TFlite Model for Face and Mask Detection instead of Caffeemodel |
| -onlyFaces | onlyFaces | False | Will skip the mask detection procedure |
Note: all arguments with multiple inputs can be specified by setting the argument again (-v vid1.mp4 -v vid2.mp4)

### Crossproduct
All crossproduct arguments get combined with each other to generate the different test runs.

ie. -n 30 -n 15 -w 100 -w 200 becomes 
* "-n 30 -w 100"
* "-n 15 -w 100"
* "-n 30 -w 200"
* "-n 15 -w 200"

# analyse.py

## Analyse the results
> python3 analyse.py --sourceFolder `"the results folder location"`

## Arguments
- Get list of input arguments by using 
> python3 analyse.py -h

| Argument  | Name | Default | Description | 
|---|---|---|---|
| -s  | sourceFolder | --- | The results folder |
