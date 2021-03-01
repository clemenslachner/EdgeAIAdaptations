import subprocess
import os
import json
import argparse
import itertools
from subprocess import PIPE
import ntpath
import sys

BASELINE_PARAMS = [('checkFrame',1)]
BASELINE_NAME = 'baseline.json'

SHORT_DICT = {
    "width": "w",
    "checkFrame": "c",
    "edge": "e",
    "gray": "g",
    "anonymization" : "a"
}

VERBOSE_MODE = True

class VideoTest:
    def __init__(self, videoPath, outputFolder):
        self.videoPath = videoPath
        self.videoName = ntpath.basename(videoPath)
        self.outputFolder = outputFolder
        if not os.path.exists(os.path.join(outputFolder, self.videoName)):
            os.makedirs(os.path.join(outputFolder, self.videoName))
        if(os.path.isfile(os.path.join(outputFolder, self.videoName,BASELINE_NAME))):
            self.hasBaseline = True
        else:
            self.hasBaseline = False
        self.runs = []
    
    def generateBaseline(self, settingArguments):
        if(self.hasBaseline):
            print(f"Skipping Baseline for {self.videoName}")
            return

        settingArguments.append(('inputVideo',self.videoPath))


        print(f"Generating Baseline for {self.videoName}")

        testResult = self.runTest(generateArgumentArray(BASELINE_PARAMS,settingArguments), 'baseline')
        testResult["shortName"] = 'baseline'
        
        f= open(os.path.join(self.outputFolder,self.videoName,BASELINE_NAME),"w+")
        f.write(json.dumps(testResult, indent=4))
        f.close()
    
    def runAllTests(self, products, settingArguments):
        print(f"Executing {len(products)} Tests for {self.videoName}")

        result = []

        settingArguments.append(('inputVideo',self.videoPath))

        for product in products:
            shortName = generateShortName(product)
            testResult = self.runTest(generateArgumentArray(product,settingArguments), shortName)
            testResult["shortName"] = shortName
            result.append(testResult)
        
        f= open(os.path.join(self.outputFolder,self.videoName,'testRun.json'),"w+")
        f.write(json.dumps(result, indent=4))
        f.close()
        
    def runTest(self, arguments, shortName=None):
        if(shortName):
            print('\tRunning ',shortName)
        else:
            print('\tRunning ',arguments)
        global VERBOSE_MODE
        if VERBOSE_MODE:
            print("\t"," ".join(arguments))
        ret = None
        proc = None
        try:
            proc = subprocess.run(arguments, stdout=PIPE, stderr=PIPE)
            with open('tmp.json') as json_file:
                ret = json.load(json_file)
            os.remove('tmp.json')
        except subprocess.CalledProcessError as e:
            print (e.output)
            if proc:
                print(proc.stdout)
        except:
            print(sys.exc_info()[0])
            if proc:
                print(proc.stdout)
        return ret
        

def generateShortName(argumentList):
    args = []
    for name, value in argumentList:
        args.append(shorten(name) + ":" + str(value))
    return ", ".join(args)

def shorten(name):
    if(name in SHORT_DICT):
        return SHORT_DICT[name]
    return name

def generateArgumentArray(testArguments, settingArguments):
    res = ["python3", "detect_mask_video.py", "--output", "tmp.json"]

    for name, value in settingArguments:
        res.append('--'+name)
        #if(value != None and value != ''):
        res.append(str(value))
    
    for name, value in testArguments:
        res.append('--'+name)
        res.append(str(value))
    return res

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def setupArguments():
    parser = argparse.ArgumentParser()

    # from detect_mask_video
    parser.add_argument("-c", "--confidence", action='append', type=float, help="minimum probability to filter weak detections")

    parser.add_argument("-w", "--width", action='append', type=int, help="resize the input video frames")
    parser.add_argument("-g", "--gray", action='append', type=boolean_string, help="convert the analyzed frames to grayscale")
    parser.add_argument("-edge", "--edge", action='append', type=boolean_string, help="preprocess each frame by sharpening edges in the video")
    parser.add_argument("-n", "--checkFrame", action='append', type=int, help="each n-th frame will be analyzed")
    parser.add_argument("-a", "--anonymization", action='append', type=int, help="0...no anoymization, 1...simpleRect, 2...pixelate, 3...gaussianBlur")

    # own params
    parser.add_argument("-i", "--videoFolder", type=str, help="the path to the videos")
    parser.add_argument("-v", "--videos", type=str, action='append', help="the path to the videos")
    parser.add_argument("-b", "--generateBaseline", type=bool, default=False, const=True, nargs='?', help="generates baselines flag")
    parser.add_argument("-o", "--outputFolder", type=str, default="results", help="the output folder")
    parser.add_argument("-verbose", "--verbose", type=bool, default=True, const=True, nargs='?', help="print extra info")

    # settings params
    parser.add_argument("-gpu", "--gpu", type=str, default="", help="device ids of gpu's to be used in the formet id,id,...")
    parser.add_argument("-useTflite", "--useTflite", type=bool, default=False, const=True, nargs='?', help="use useTflite for face detection instead of caffeemodel")
    parser.add_argument("-useTf", "--useTf", type=bool, default=False, const=True, nargs='?', help="use TF model for face detection instead of caffeemodel")
    parser.add_argument("-onlyFaces", "--onlyFaces", type=bool, default=False, help="ignore masks on TF-model and only search for faces")
    parser.add_argument("-d", "--display", type=bool, default=False, const=True, nargs='?', help="display video while analysing")
    
    return parser

def getTestArguments(args):
    testArgs = []
    for (key, values) in args.items():
        if key == 'width' or key == 'gray' or key == 'checkFrame' or key == 'edge' or key == 'anonymization':
            if values != None:
                tmp = []
                values = values if type(values) in [list, tuple] else [values]
                for value in values:
                    tmp.append((key,value))
                testArgs.append(tmp)
    return testArgs

def getSettingArguments(args):
    settingArgs = []
    if args['gpu'] != None and args['gpu'] != "":
        settingArgs.append(('gpu',args['gpu']))
    if args['useTflite']:
        settingArgs.append(('useTflite','True'))
    if args['useTf']:
        settingArgs.append(('useTf','True'))
    if args['onlyFaces']:
        settingArgs.append(('onlyFaces','True'))
    if args['display']:
        settingArgs.append(('display','True'))
    return settingArgs

def main():
    parser = setupArguments()
    args = vars(parser.parse_args())

    global VERBOSE_MODE
    VERBOSE_MODE = args["verbose"]

    testArgs = getTestArguments(args)
    settingArgs = getSettingArguments(args)

    print(f"Running Tests with {len(settingArgs)} Extra settings:")
    for setting in settingArgs:
        print("\t",setting)

    if not os.path.exists(args['outputFolder']):
        os.makedirs(args['outputFolder'])

    videos = []

    if(args['videoFolder'] != None and args['videos'] != None):
        print('Cannot use videoFolder and videos at the same time')
        return
    elif(args['videoFolder'] != None):
        videos = [os.path.join(args['videoFolder'], file) for file in os.listdir(args['videoFolder']) if not (file.endswith(".json") or file.endswith(".png"))]
    elif(args['videos'] != None):
        videos = args['videos']

    if(len(videos) == 0):
        print('No Videos specified!')
        return

    videoTests = [VideoTest(video,args['outputFolder']) for video in videos]

    if(args['generateBaseline']):
        print(f"Generating {len(videoTests)} Baselines")
        
        for videoTest in videoTests:
            videoTest.generateBaseline(settingArgs)
        return

    products = list(itertools.product(*testArgs))

    print('Generated '+str(len(products))+' version(s) of input arguments')

    print("Found {:d} video(s)".format(len(videos)))

    for videoTest in videoTests:
        videoTest.runAllTests(products, settingArgs)

if __name__ == "__main__":
    main()
