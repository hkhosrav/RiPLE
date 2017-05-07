
# coding: utf-8

# In[1]:

import numpy as np
import scipy.stats
import csv
import os

#settings
KGw=1
beta=0.1

inputfolder = 'input'
recomfolder = 'recoms'
recommender='BiasedMatrixFactorization'
fullsetFile = 'fullset.csv'
predictionFile = 'predictions.csv'

QTFile ="QT.csv"
SQFile = "SQ.csv"
RFile = "R.csv"
# install MyMediaLite and point the EXEFILE to the rating_prediction.exe file
#EXEFILE = '/Users/hassankhosravi/Dropbox/UQ/Projects/RecSysTEL/code/Libaries/MyMediaLite-3.11/lib/mymedialite/rating_prediction.exe'
EXEFILE = '/Users/uqhkhosr/Dropbox/UQ/Projects/RecSysTEL/code/Libaries/MyMediaLite-3.11/lib/mymedialite/rating_prediction.exe'


# # Loading Input Data

# In[2]:

# Returns a Dict that maps content of pathName/fileName/colNum to an index 
def mapContentToIndex(pathName, fileName, colNum):
    qfile = pathName + "/" + fileName
    csv_file = csv.reader(open(qfile, "rU"), delimiter=",")
    csv_file.next()
    map = {}
    current =0
    for row in csv_file:
        tag = row[colNum]
        if map.has_key(tag)==False:
            map[tag]=current
            current = current+1
    return map

# Returns a matrix Question by Tags where QTMat[i][j] is 1/g if i is tagged with g topics, including j and 0 otherwise
def createTMatrix(pathName, fileName, qDict, qSize, tDict, tSize):
    QTMat = np.zeros((qSize, tSize))
    qfile = pathName + "/" + fileName
    csv_file = csv.reader(open(qfile, "rU"), delimiter=",")
    # csv_file.next() add this line if file has header    
    for row in csv_file:
        qid = row[0]
        tag = row[1] 
        if qDict.has_key(qid)==True and tDict.has_key(tag)==True:
            QTMat[qDict[qid]][tDict[tag]] =1
        
    sumRows = QTMat.sum(axis=1)
    for i in range (1, len(sumRows)):
        if sumRows[i]>0:
            QTMat[i] = QTMat[i]/sumRows[i]
    return QTMat


# Loads file at pathName/FileNAme/ColNum into a matrix using uDict and qDict
def load(pathName, fileName, colNum, uDict, uSize, qDict, qSize, delimiter ):
    matrix = np.zeros((uSize, qSize))
    qfile = pathName + "/" + fileName
    csv_file = csv.reader(open(qfile, "rU"), delimiter=delimiter)
    csv_file.next() #add this line if file has header    
    i =0;
    for row in csv_file:
        id1 = row[0]
        id2  = row[1] 
        value = row[colNum]
        if uDict.has_key(id1)==True and qDict.has_key(id2)==True:
            matrix[uDict[id1]][qDict[id2]] = value
    return matrix


#returns 1 for non zero cells
def createIndexMatrix(M):
    I_M = M.copy()
    I_M[I_M > 0] = 1
    I_M[I_M < 0] = 1
    I_M[I_M == 0] = 0
    return I_M


def loadDataset(inputfolder):
    tDict = mapContentToIndex(inputfolder, QTFile,1)
    tSize = len(tDict)
    uDict = mapContentToIndex(inputfolder,SQFile,0)
    uSize = len(uDict)
    qDict = mapContentToIndex(inputfolder, SQFile,1)
    qSize = len(qDict)
    T =  createTMatrix(inputfolder, QTFile, qDict, qSize, tDict, tSize)
    A  = load(inputfolder, SQFile, 2, uDict, uSize, qDict, qSize,',')
    I_A = createIndexMatrix(A)
    D  = load(inputfolder,SQFile, 3, uDict, uSize, qDict, qSize,',')
    D = D/D.max() #normalize
    I_D  = createIndexMatrix(D)
    P  = load(inputfolder, SQFile, 4, uDict, uSize, qDict, qSize,',')
    P = P/P.max()
    I_P  = createIndexMatrix(P)
    return A, I_A, D, I_D, P, I_P, T, uDict, uSize, qDict, qSize, tDict, tSize        




# # Data Integration

# In[3]:

# takes in a Matrix and returns a vector with mean of non zero values per column
def computeNonZeroMean(M): 
    temp = M.copy()
    temp[np.where(temp == 0)] = np.nan
    x = np.nanmean(temp,axis=0)
    y = np.nanstd(temp, axis=0)
    return np.nan_to_num(x), np.nan_to_num(y)


#computing knowledge gap per question. negative values indicate competencies
def computeKnowledgeGap(A, D, I_D):
    B = np.zeros((len(A), len(A[0])))
    D_mean, D_std = computeNonZeroMean(D)
    for i in range(0, len(A)):
        for j in range(0 , len(A[0])):
            if I_D[i][j]>0:
                #print (1- A[i])*(0.5 - A[i])/( D[i]) + A[i]* (0.5 - A[i])/(1 - D[i])  
                firstPart = (1- A[i][j])*(0 - A[i][j])/(1 + D_mean[j]) # contributes when answered incorrectly
                secondPart =(1+A[i][j])* (0 - A[i][j])/(2 - D_mean[j])  # contributes when answered correctly
                B[i][j] =   firstPart + secondPart 
    
    return B

# writes matrix R into PathName/FileName
def WriteMatrixToTable(PathName, FileName, R, uDict, qDict, headings):
    if not os.path.exists(PathName):
        os.makedirs(PathName)
        
    FilePath = PathName + "/" + FileName
    fr = open(FilePath, 'w')
    try:
        writer = csv.writer(fr)
        if headings:
            writer.writerow(headings)
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] <> 0:
                    writer.writerow([uDict.keys()[uDict.values().index(i)], qDict.keys()[qDict.values().index(j)], R[i][j]])
                    # writer.writerow([i, j, R[i][j]])
    finally:
        fr.close() 
        
def combineKnowledgeGapPreference(A, D, P, I_P, I_D, KGw):
    KG = computeKnowledgeGap(A,D,I_D)
    
    R = (KGw * KG + (1-KGw)*P) * I_P * I_D
    return R        

#creating a full testset to use the RecSys for initial prediction
def createFullTestset(uDict, uSize, qDict, qSize):
    testset = []
    for ukey in uDict:
        for qkey in qDict:
            testset.append([ukey, qkey, 0])
    return testset

# writes listname to PathName/Filename using headings
def createFile(PathName, FileName, listname, headings):
    if not os.path.exists(PathName):
        os.makedirs(PathName)
    
    FilePath = PathName + "/" + FileName
    fr = open(FilePath, 'w')
    try:
        writer = csv.writer(fr)
        if headings:
            writer.writerow(headings)
        for item in listname:
            #Write item to outcsv
            output=','.join(str(x) for x in item)
            writer.writerow(item)
    finally:
        fr.close()

#
def LoadingR(A, D, P, I_P, I_D, uDict, uSize, qDict, qSize, KGw):
    R = np.round(combineKnowledgeGapPreference(A, D, P, I_P, I_D, KGw),2) #generate R
    I_R = createIndexMatrix(R)
    WriteMatrixToTable(recomfolder, RFile, R, uDict, qDict,[])
    fullset = createFullTestset(uDict, uSize, qDict, qSize)
    createFile(recomfolder, fullsetFile, fullset, [])
    return  R, I_R

A, I_A, D, I_D, P, I_P, T, uDict, uSize,  qDict, qSize, tDict, tSize = loadDataset(inputfolder)
R, I_R = LoadingR(A, D, P, I_P, I_D, uDict, uSize, qDict, qSize, KGw) #generate R   


# In[ ]:




# # Learning Profile

# In[4]:

# creates the learning profile
def LearingProfile(R,T, I_R):
    impact = np.dot(R,T)
    weight = np.dot(I_R, T)+1 # to avoid division by zero
    LP = impact/weight
    return LP



# # Recommendation Engine
# ## Traditional recommendation

# In[5]:

from subprocess import check_output

def checkCreate(inputfolder):
    directory = os.getcwd() + '/' + inputfolder
    if not os.path.exists(directory):
        os.makedirs(directory)


def  RecSys(inputfolder, recomfolder, recommender, trainset, testset, output):
    checkCreate(recomfolder)
    executable = EXEFILE
    check_output(
                ['mono', executable,
                '--training-file', recomfolder + '/' + trainset,
                '--test-file', recomfolder + '/'+ testset,
                '--recommender', recommender,            
                '--prediction-file' , recomfolder + '/' + output])

RecSys(inputfolder, recomfolder, recommender,RFile, fullsetFile, predictionFile)


# ## Updating and Enhancing Recommendations

# In[6]:

# Ehhancing the predictions from RecSysTel using the learning profile
def updateRecommendation(R, LP, T,beta):
    H = np.dot(LP, T.T)
    Rprime = R + H*beta
    return Rprime    


# # Recommendation 

# In[11]:

def generateRecommendations(KGw, beta):
    # Load Data set
    A, I_A, D, I_D, P, I_P, T, uDict, uSize,  qDict, qSize, tDict, tSize = loadDataset(inputfolder)
    R, I_R = LoadingR(A, D, P, I_P, I_D, uDict, uSize, qDict, qSize, KGw) #generate R   

    # creates learning profile 
    LP = LearingProfile(R,T, I_R)

    # Run Recommender System
    RecSys(inputfolder, recomfolder, recommender,RFile, fullsetFile, predictionFile)

    # Ehhancing the predictions from RecSysTel using the learning profile
    Rprime = load(recomfolder, predictionFile,2, uDict, uSize, qDict, qSize, "	")
    O = updateRecommendation(Rprime, LP, T,beta)    
    return O

# generated recommendations
O = generateRecommendations(KGw, beta)
    
    
  


# In[12]:

# 
def recommendation(user, O):
    row = uDict[user]
    values = O[row].tolist() # recommendations
    maxQuestionIndex = values.index(max(values)) #index of recom with highest value
    qid = qDict.keys()[qDict.values().index(maxQuestionIndex)] #question of the recommended topic
    return qid  

for key in uDict:
	print key, recommendation(key, O)





