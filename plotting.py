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
            if pair[1] > pareto_front[-1][1]: # before was >=
                pareto_front.append(pair)
        else:
            if pair[1] < pareto_front[-1][1]: # before was <=
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
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid()
    plt.savefig(os.path.join(plotPath, title) + ".png")
    plt.show()

def plotLatencyVsAccuracyAllModels(dpuList, latencyData, accuracyData, tot=False):
    columnsList = ["224","192","160","128"]
    indexList = ["1.0","0.75","0.5","0.25"]
    Xs = []
    Ys = []
    labels = []
    for dpu in dpuList:
        for alpha in indexList:
            for imageSize in columnsList:
                latency = latencyData[dpu].loc[alpha, imageSize]
                if not np.isnan(latency):
                    Xs.append(getFPS(latency))
                    Ys.append(accuracyData.loc[alpha, imageSize])
                    labels.append((dpu, alpha, imageSize))
    title = f" All models"
    plot_pareto_frontier(title, Xs, Ys, labels, yLabel="Accuracy", xLabel="FPS")

# def plotLatencyVsAccuracyAllModels(dpuList, latencyData, accuracyData, tot=False):
#     columnsList = ["224","192","160","128"]
#     indexList = ["1.0","0.75","0.5","0.25"]
#     maxX=True
#     maxY=True

#     b = bPlotSize
#     a = b*(6.4/4.8)
#     plt.figure(figsize=(a,b))
    
#     for dpu in dpuList:
#         Xs = []
#         Ys = []
#         labels = []
#         for alpha in indexList:
#             for imageSize in columnsList:
#                 latency = latencyData[dpu].loc[alpha, imageSize]
#                 if not np.isnan(latency):
#                     Xs.append(getFPS(latency))
#                     Ys.append(accuracyData.loc[alpha, imageSize])
#                     labels.append((dpu, alpha, imageSize))

#         '''Pareto frontier selection process'''
#         sorted_list = sorted([[Xs[i], Ys[i], labels[i]] for i in range(len(Xs))], reverse=maxY)
#         pareto_front = [sorted_list[0]]
#         for pair in sorted_list[1:]:
#             if maxY:
#                 if pair[1] >= pareto_front[-1][1]:
#                     pareto_front.append(pair)
#             else:
#                 if pair[1] <= pareto_front[-1][1]:
#                     pareto_front.append(pair)
        
#         '''Plotting process'''
#         plt.scatter(Xs,Ys)
#         pf_X = [pair[0] for pair in pareto_front]
#         pf_Y = [pair[1] for pair in pareto_front]
#         pf_labels = [pair[2] for pair in pareto_front]
#         plt.plot(pf_X, pf_Y)
#         for x,y,label in zip(pf_X, pf_Y, pf_labels):
#             plt.text(x, y, label)

#     plt.xlabel("FPS")
#     plt.ylabel("Accuracy")
#     plt.title("All models")
#     plt.grid()
#     plt.savefig(os.path.join(plotPath, "All models 2") + ".png")
#     plt.show()


def plotLatencyVsAccuracy(dpu, latencyData, accuracyData, tot=False):
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
    # if tot:
    #     title = dpu + " (preprocessing + inference time)"
    # else:
    #     title = dpu + " (inference time)"
    title = dpu
    plot_pareto_frontier(title, Xs, Ys, labels, yLabel="Accuracy", xLabel="FPS")

def plotLUTvsFPS(dpuList, latencyData, lutData):
    plt.figure()
    for alpha in [1.0, 0.75, 0.5, 0.25]:   
        Ys = []
        Xs = [] 
        for dpu in dpuList:
            latency = latencyData[dpu].loc[str(alpha), str(224)]
            if not np.isnan(latency):
                Ys.append(getFPS(latency))            
            Xs.append(int(lutData[dpu]))
        plt.plot(Xs, Ys, label=f"{alpha} 224", marker="o", linestyle="dotted")
        
    plt.legend()
    plt.grid()
    plt.ylabel("FPS")
    plt.xlabel("Lookup Table")
    plt.savefig(os.path.join(plotPath, "FPSvsLUT.png"))
    plt.show()

def plotBRAMvsFPS(dpuList, latencyData, BRAMData):
    plt.figure()
    for alpha in [1.0, 0.75, 0.5, 0.25]:   
        Ys = []
        Xs = [] 
        for dpu in dpuList:
            latency = latencyData[dpu].loc[str(alpha), str(224)]
            if not np.isnan(latency):
                Ys.append(getFPS(latency))            
            Xs.append(int(BRAMData[dpu]))
        plt.plot(Xs, Ys, label=f"{alpha} 224", marker="o", linestyle="dotted")
        
    plt.legend()
    plt.grid()
    plt.ylabel("FPS")
    plt.xlabel("BRAM")
    plt.savefig(os.path.join(plotPath, "FPSvsBRAM.png"))
    plt.show()

def plotBarChartAcc(accuracyGoogleData, accuracyData):
    imageSizeList = ["224","192","160","128"]
    alphaList = ["1.0","0.75","0.5","0.25"]

    googleAccList = []
    vaiAccList = []
    columnsLabels = []
    for alpha in alphaList:
        for imageSize in imageSizeList:
            googleAccList.append(accuracyGoogleData.loc[alpha, imageSize])
            vaiAccList.append(accuracyData.loc[alpha, imageSize])
            columnsLabels.append(f"({alpha},{imageSize})")
    
    x_axis = np.arange(len(columnsLabels))
    b = bPlotSize
    a = b*(6.4/4.8)
    plt.figure(figsize=(a,b))
    plt.bar(x_axis + 0.2, vaiAccList, width=0.4, label="Vitis AI PTQ") # tick_label=f"{alpha}, {imageSize}"
    plt.bar(x_axis - 0.2, googleAccList, width=0.4, label="Google ATQ")
    plt.xticks(x_axis, columnsLabels, fontweight='bold')
    plt.legend(fontsize=20)
    plt.title("Post-training quantization vs quantization aware-training", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.savefig(os.path.join(plotPath, "VAIvsGOOGLE_accuracy.png"))
    plt.show()

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
    # nameList = ["B4096_latency.csv", "B3136_latency.csv", "B2304_latency.csv", "B1600_latency.csv", "B1152_latency.csv", "B1024_latency.csv", "B800_latency.csv", "B512_latency.csv"]
    nameListTot = ["B4096_tot_latency.csv", "B3136_tot_latency.csv", "B2304_tot_latency.csv", "B1600_tot_latency.csv", "B1152_tot_latency.csv", "B1024_tot_latency.csv", "B800_tot_latency.csv", "B512_tot_latency.csv"]

    dpuList = []
    for name in nameListTot:
        dpuList.append(name.split("_")[0])
    accuracyFileName = "accuracyQuantizedModels.csv"
    accuracyGoogleFileName = "accuracyGoogleQuantizedModels.csv"
    lutFileName = "LUTperDPU.csv"
    BRAMFileName = "BRAMperDPU.csv"
    global plotPath
    plotPath = os.path.join("PlotData", "Plots") 

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpu", type=str, default="B4096", choices=dpuList)
    parser.add_argument("--plotSize", type=float, default=5.2)
    parser.add_argument("--acc", action='store_true')
    parser.add_argument("--lut", action='store_true')
    parser.add_argument("--tot", action='store_true')
    parser.add_argument("--bram", action='store_true')
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bar", action="store_true")    
    args = parser.parse_args()

    global bPlotSize
    bPlotSize = args.plotSize

    print("************************************")
    print("INPUT PARAMETERS:")
    print(f"\tDPU: {args.dpu}")
    print(f"\tPlot Accuracy vs FPS: {args.acc}")
    print(f"\tPlot LUT vs FPS: {args.lut}")
    print("************************************")

    accuracyData = getAccuracyData(accuracyFileName)        
    accuracyGoogleData = getAccuracyData(accuracyGoogleFileName)
    latencyData = getLatencyData(nameListTot)
    lutData = getLutData(lutFileName)
    BRAMData = getLutData(BRAMFileName)

    if args.all:
        plotLatencyVsAccuracyAllModels(dpuList, latencyData, accuracyGoogleData, tot=False)

    if args.acc:
        plotLatencyVsAccuracy(args.dpu, latencyData, accuracyGoogleData, tot=True)    

    if args.lut:        
        plotLUTvsFPS(dpuList, latencyData, lutData) 

    if args.bram:
        plotBRAMvsFPS(dpuList, latencyData, BRAMData) 

    if args.bar:
        plotBarChartAcc(accuracyGoogleData, accuracyData)

    

    # printAllLatencyData(latencyData, nameList)
    # print(accuracyData)
    # print(accuracyData.loc["1.0", "224"])
    # fps = getFPS(latencyData["B4096"].loc["1.0","224"])
    # print(fps)

if __name__ == "__main__":
    main()
