import csv
from ntpath import join
import os
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

"""
./docker_run.sh xilinx/vitis-ai:1.3.411
"""

"""
Xs: Latency
Ys: Accuracy
"""
def plot_pareto_frontier(title, Xs, Ys, labels, yLabel, xLabel, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i], labels[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    '''Plotting process'''
    b = bPlotSize
    a = b*(6.4/4.8)
    plt.figure(figsize=(a,b))
    plt.scatter(Xs,Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    pf_labels = [pair[2] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)
    for x,y,label in zip(pf_X, pf_Y, pf_labels):
        plt.text(x, y, label)
    plt.xlabel("FPS")
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid()
    plt.savefig(os.path.join(plotPath, title) + ".png")
    plt.show()

# def plotLUTvsAccuracy(dputList, latencyData, accuracyData, lutData):
#     columnsList = ["224","192","160","128"]
#     indexList = ["1.0","0.75","0.5","0.25"]
#     Xs = []
#     Ys = []
#     labels = []
#     for dpu in dputList:
#         for alpha in indexList:
#             for imageSize in columnsList:
#                 latency = latencyData[dpu].loc[alpha, imageSize]
#                 if not np.isnan(latency):
#                     Xs.append(int(lutData[dpu]))
#                     Ys.append(accuracyData.loc[alpha, imageSize])
#                     labels.append((alpha, imageSize))
#     title = f"LUT vs Accuracy"
#     plot_pareto_frontier(title, Xs, Ys, labels, yLabel="Accuracy", xLabel="LUT")

def plotLatencyVsAccuracy(dpu, latencyData, accuracyData):
    columnsList = ["224","192","160","128"]
    indexList = ["1.0","0.75","0.5","0.25"]
    Xs = []
    Ys = []
    labels = []
    for alpha in indexList:
        for imageSize in columnsList:
            latency = latencyData[dpu].loc[alpha, imageSize]
            if not np.isnan(latency):
                Xs.append(getFPS(latency))
                Ys.append(accuracyData.loc[alpha, imageSize])
                labels.append((alpha, imageSize))
    plot_pareto_frontier(dpu, Xs, Ys, labels, yLabel="Accuracy", xLabel="FPS")

def plotLUTvsFPS(alpha, imageSize, dpuList,latencyData, lutData):
    # print(int(lutData["B4096"]))
    Xs = []
    Ys = []
    labels = []
    for dpu in dpuList:
        latency = latencyData[dpu].loc[str(alpha), str(imageSize)]
        if not np.isnan(latency):
            Xs.append(getFPS(latency))
            Ys.append(int(lutData[dpu]))
            labels.append(dpu)
    title = f"mobilenet_v1_{alpha}_{imageSize}"
    plot_pareto_frontier(title, Xs, Ys, labels, yLabel="LUT", xLabel="FPS")

# def plotLUTvsFPS_alternative(alpha, imageSize, dpuList,latencyData, lutData):
#     # print(int(lutData["B4096"]))
#     columnsList = ["224","192","160","128"]
#     indexList = ["1.0","0.75","0.5","0.25"]
#     Xs = []
#     Ys = []
#     labels = []
#     for dpu in dpuList:
#         for alpha in indexList:
#             for imageSize in columnsList:
#                 latency = latencyData[dpu].loc[str(alpha), str(imageSize)]
#                 if not np.isnan(latency):
#                     Xs.append(getFPS(latency))
#                     Ys.append(int(lutData[dpu]))
#                     labels.append(dpu)
#     title = f"boh"
#     plot_pareto_frontier(title, Xs, Ys, labels, yLabel="LUT", xLabel="FPS")

def readCsv(dataPath):
    with open(dataPath, newline="") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        matrix =  np.full((4,4), -1 , dtype=np.float64)
        for i, row in enumerate(csvReader):
            matrix[i][0] = row[0]
            matrix[i][1] = row[1]
            matrix[i][2] = row[2]
            matrix[i][3] = row[3]

    return matrix

def printAllLatencyData(latencyData, nameList):
    for name in nameList:
        print(name.split("_")[0])
        print(latencyData[name.split("_")[0]])

def getLatencyData(nameList):
    latencyData = {}
    for name in nameList:
        dataPath = os.path.join("PlotData", name)
        latencyMatrix = readCsv(dataPath)
        dataFrame = pd.DataFrame(latencyMatrix, columns=["224","192","160","128"], index=["1.0","0.75","0.5","0.25"])
        latencyData[name.split("_")[0]] = dataFrame

    return latencyData

def getAccuracyData(fileName):
    dataPath = os.path.join("PlotData", fileName)
    accuracyMatrix = readCsv(dataPath)
    dataFrame = pd.DataFrame(accuracyMatrix, columns=["224","192","160","128"], index=["1.0","0.75","0.5","0.25"])
    
    return dataFrame

def getLutData(lutFileName):
    dataPath = os.path.join("PlotData", lutFileName)

    return pd.read_csv(dataPath)

def getFPS(latency):
    fps = 1 / latency
    return int(fps)

def main():
    nameList = ["B4096_latency.csv", "B3136_latency.csv", "B2304_latency.csv", "B1600_latency.csv", "B1152_latency.csv", "B1024_latency.csv", "B800_latency.csv", "B512_latency.csv", ]
    dpuList = []
    for name in nameList:
        dpuList.append(name.split("_")[0])
    accuracyFileName = "accuracyQuantizedModels.csv"
    lutFileName = "LUTperDPU.csv"
    global plotPath
    plotPath = os.path.join("PlotData", "Plots") 

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imageSize", type=int, default=224)
    parser.add_argument("-a", "--alpha", type=float, default=1.0)
    parser.add_argument("-d", "--dpu", type=str, default="B4096")
    parser.add_argument("-s", "--plotSize", type=float, default=5.2)
    parser.add_argument("-l", "--lut", action='store_true')
    args = parser.parse_args()

    global bPlotSize
    bPlotSize = args.plotSize

    if not args.lut:
        print("************************************")
        print("INPUT PARAMETERS:")
        print(f"\tDPU: {args.dpu}")
        print(f"\tPlot Accuracy vs FPS: {not args.lut}")
        print(f"\tPlot LUT vs FPS: {args.lut}")
        print("************************************")

        latencyData = getLatencyData(nameList)
        accuracyData = getAccuracyData(accuracyFileName)
        plotLatencyVsAccuracy(args.dpu, latencyData, accuracyData)

    if args.lut:
        print("************************************")
        print("INPUT PARAMETERS:")
        print(f"\tAlpha: {args.alpha}")
        print(f"\tImage Size: {args.imageSize}")
        print(f"\tPlot Accuracy vs FPS: {not args.lut}")
        print(f"\tPlot LUT vs FPS: {args.lut}")
        print("************************************")

        latencyData = getLatencyData(nameList)
        lutData = getLutData(lutFileName)
        accuracyData = getAccuracyData(accuracyFileName)
        plotLUTvsFPS(args.alpha, args.imageSize, dpuList, latencyData, lutData) 
        # plotLUTvsAccuracy(dpuList, latencyData, accuracyData, lutData)

    # printAllLatencyData(latencyData, nameList)

    # print(accuracyData)


    # print(accuracyData.loc["1.0", "224"])
    # fps = getFPS(latencyData["B4096"].loc["1.0","224"])
    # print(fps)

if __name__ == "__main__":
    main()
