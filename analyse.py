import argparse
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import numpy as np
import json
import os

BASELINE_NAME = 'baseline.json'

markers=["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "+", "D"]

def setupArguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--sourceFolder", type=str, default="results", help="The results folder")
    return parser

def loadSourceFolder(folderPath):
    source = {}
    baselines = {}

    for root, dirs, files in os.walk(folderPath):
        for video in dirs:
            with open(os.path.join(folderPath,video,'testRun.json')) as json_file:
                rawJson = json.load(json_file)
                source[video] = {}
                for singleTest in rawJson:
                    source[video][singleTest['shortName']] = singleTest
            with open(os.path.join(folderPath,video,BASELINE_NAME)) as json_file:
                baselines[video] = json.load(json_file)

    return source, baselines

def main():
    parser = setupArguments()
    args = vars(parser.parse_args())

    source, baselines = loadSourceFolder(args['sourceFolder'])

    for video in baselines:
        source[video]['baseline'] = baselines[video]
    
    generateFPSBarPlot(args, source, args['sourceFolder'])

    generateLinePlot(args, source,"accuracy", args['sourceFolder'])
    generateLinePlot(args, source,"faces", args['sourceFolder'])
    generateLinePlot(args, source,"masks", args['sourceFolder'])
    generateTimingsBarPlot(args, source, args['sourceFolder'])
    
def generateLinePlot(args,source, metric, sourceFolder):
    print("Generating "+metric+" Plot")

    for video in source:
        # Clearing Plot
        plt.clf()
        #plt.figure(dpi=1200)

        videoDict = source[video]
        markerIndex=0
        print("Video: ", video)
        for version in videoDict:
            versionDict = videoDict[version]
            x = [frame["frameId"] for frame in versionDict["frames"]]
            y = [frame["frameMetrics"][metric] for frame in versionDict["frames"]]
            if(version == 'baseline'):
                plt.plot(x, y, ':', label=version)
            else:
                plt.plot(x, y, markers[markerIndex]+"--", label=version, alpha=0.5)
                if (markerIndex >= len(markers)-1):
                   markerIndex=0
                else:
                   markerIndex=markerIndex+1
        plt.ylabel(metric)
        plt.xlabel("frames")
        plt.title(video + " " + metric)
        legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1))
        pngPath = os.path.join(sourceFolder,video,metric+".svg")
        plt.savefig(pngPath, bbox_extra_artists=(legend,), bbox_inches='tight')

def generateFPSBarPlot(args, source, sourceFolder):   
    print("Generating FPS Barplot")
    for videoName, curVideo in source.items():
        # Clearing Plot
        plt.clf()
        print("Video: ", videoName)

        labels = [version.replace("), (",")\n(").replace("width","w").replace("checkFrame","c") for version in curVideo]
        values = [val["fps"] for _, val in curVideo.items()]
        
        for i in range(len(values)):
            plt.bar("V{}".format(i), values[i])
        plt.ylabel('fps processed')
        plt.xlabel('Versions')
        plt.title(videoName + " FPS")

        legend = plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
        pngPath = os.path.join(sourceFolder,videoName,"fps.svg")
        plt.savefig(pngPath, bbox_extra_artists=(legend,), bbox_inches='tight')

def generateTimingsBarPlot(args, source, sourceFolder):
    print("Generating Timings Barplot")
    for video in source:
        # Clearing Plot
        plt.clf()
        print("Video: ", video)

        videoDict = source[video]
        avgTimings = {}
        #(str(videoDict))

        for version in videoDict:
            if(version != 'baseline'):
                versionDict = videoDict[version]
                timings = [frame["frameMetrics"]["timings"] for frame in versionDict["frames"]]
                # calculate sum of timings for each recorded step
                nrTimings = len(timings)
                sumTimings = {}
                print("Version: " + str(version))
                # print('Number of timings {:d}'.format(nrTimings))
                for dict in timings: 
                    for list in dict: 
                        if list in sumTimings: 
                            sumTimings[list] += (dict[list]) 
                        else: 
                            sumTimings[list] = dict[list] 
            
                # print("The summed timings for each step: " + str(sumTimings))
                # calculate avg timing for each recorded step timing
                for step in sumTimings:
                    stepAsString = str(step)
                    avg = sumTimings[step] / nrTimings
                    if step == "completeStep" or step == "inferencing-tf":
                        avgTimings[step] = avg 
                    else:
                        avgTimings[version] = avg
            
                # print("The average timings for each step: " + str(sumTimings))
                # print("Length: " + str(len(sumTimings)))
        print(avgTimings)
        for key, val in avgTimings.items():
            # print("Key: " + str(key))
            # print("Value: " + str(val))
            plt.bar(key, val*1000)
            plt.ylabel('time in ms')
            plt.xlabel('Frame Processing Step')
            plt.title(video + "Average Timings")

        # legend = plt.legend(sumTimings.keys(), loc='upper center', bbox_to_anchor=(0.5,-0.1))
        pngPath = os.path.join(sourceFolder,video,"timings.svg")
        plt.savefig(pngPath, bbox_inches='tight')

if __name__ == "__main__":
    main()