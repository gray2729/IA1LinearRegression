"""
File name:  LinearRegression.py
Author:     Isaac Gray
Date:  	    02/12/2023
Class: 	    DSCI 440 ML
Assignment: IA 1
Purpose:    This program takes a data set and finds the weight vector 
            optimizes the Sum on Square Error.
"""
import numpy as np
import matplotlib.pyplot as pyplot


"""
Function:    fDeterminew
Description: determines vector w through normal equations
Input:       X - matrix holding the features
             y - vector holding the true values of the training set
Output:      w - vector holding the parameters for the 
                 linear function
"""
def fDeterminew(X, y):
    XTranspose = np.transpose(X)
    XInverse = np.linalg.inv(np.matmul(XTranspose, X))
    
    w = np.matmul(XTranspose, y)
    w = np.matmul(XInverse, w)
    return w



"""
Function:    fDetermineSSE
Description: determines the Sum of Square Error
Input:       X - matrix holding the features
             y - vector holding the true values of the training set
             w - vector holding the parameters for the 
                 linear function
Output:      SSE - value of objective function J(w)
"""
def fDetermineSSE(X, y, w):
    wInverse = np.transpose(w)
    runningSum = 0
    for index in range(y.size):
        runningSum += (y[index] - np.matmul(wInverse, X[index]))**2
    SSE = (1/2)*runningSum
    return (SSE)



"""
Function:    main
Description: Opens and retieves the data from file, finds the weight vector w 
             and Sum of Squares Error SSE, plots the regression line and 
             training data
Input:       None
Output:      None
"""

celsiusData = []
fahrenheitData = []

fileName = "cf.txt"

file = open(fileName, "r")

lineNum = 0
for line in file:
    line = float(line.strip())
    if lineNum % 2 == 0:
        celsiusData.append([1, line])
    else:
        fahrenheitData.append(line)
    lineNum += 1

file.close()

CelsiusData = np.array(celsiusData)
FahrenheitData = np.array(fahrenheitData)

w = fDeterminew(CelsiusData, FahrenheitData)
SSE = fDetermineSSE(CelsiusData, FahrenheitData, w)

print("Optimal w: ", w)
print("SSE: ", SSE)

x = range(0,100)
y = w[0] + w[1]*x

pyplot.scatter(CelsiusData[:,1], FahrenheitData, label='training data')
pyplot.plot(x,y, c='red', label='predicted values')
pyplot.xlabel('Celsius')
pyplot.ylabel('Fahrenheit')
pyplot.title('Linear Regression')
pyplot.legend()
pyplot.show()

xInput = 33
print("Predicted value when x = 33: ", w[0] + w[1]*xInput)