
# coding: utf-8

# ## Configuration

# In[1]:

import numpy as np
from random import *
seed(123456) # a random seed used for reproducibility 

import csv
import math
import os


#Data set setting
inputfolder = 'input'

N = 400 # number of users
M= 1000 # number of questions
A= 20000 #number of answers
Q_sigma = 5 # Standard deviation on number of questions answered by each user
numCategories = 7 # number of tags
maxTopic = 2 # Maximum number of topics per question
MINRATING = 1
MAXRATING = 5

Q_mu = A/N #average number of questions per users

D_mu = 2.5 # average difficulty of generated questions
D_sigma = 2 # standard deviation of difficulty of generaetd questions
R_mu = 2.5 # average rating of generated questions
R_sigma = 2 # standard deviation of rating of generaetd questions

alpha = 0.5 # peakiness


# ## Generating synthetic dataset

# In[2]:

'''
Draw samples from Dirichlet distribution, round the probability to two decimals,
change the last one to make sure all the probabilities add up to 1.
'''
def generateUserCompetencies(knowledgeGap):
    probs = np.random.dirichlet(knowledgeGap)
    probsRounded = np.round(probs,2)
    probsRounded[-1] = 1 - np.sum(probsRounded[0:-1])
    return probsRounded

'''
Get one peaked probability from Dirichlet distribution. 
'''
def generatePeakedProbabilities(numCategories, alpha):
    probs = np.random.dirichlet(alpha=[alpha]*numCategories)
    return probs

#generatePeakedProbabilities()


'''
Draw random samples from a normal (Gaussian) distribution.
return min or max if the sample is beyond the bounds
'''
def NormalDistrubtion(Mu, Sigma, min, max):
    generated = int(np.random.normal(Mu, Sigma, 1)[0])
    if generated < min:
        return min
    elif generated > max:
        return max
    return generated
#NormalDistrubtion(Q_mu, Q_sigma, 1, 100)

'''
Get a copy of topic list so that the original one will not be changed.
Then get an array with length of max topic number per question and filled in one.
Then deciding how many topics per question randomly.
For the chosen topic, add it to the topic list and remove from the copy list.
'''
def getTopicfromDistribution(numCategories, maxTopic,listOfTopics):
    tempTopics = listOfTopics[:] # getting a copy to change
    Q = np.ones(maxTopic)*1/maxTopic
    numOfTopics = np.random.choice(np.arange(1, maxTopic+1), p=Q)
    topics = []
    for i in range(0,numOfTopics):
        P = np.ones(len(tempTopics))*1/len(tempTopics)
        choice = np.random.choice(tempTopics, p=P)
        topics.append(choice)
        tempTopics.remove(choice)                                
                                                                        
    return topics

'''
Return a list of students, where every student has an id, the number of
the questions he answers, knowleadge gaps(range 0-1) and the interest(range 0-1).
'''
def createUsers(N, T):
    users = []
    for i in range(N):
        current = []
        current.append('u' + str(i)) #id
        knowledgeGaps = generatePeakedProbabilities(numCategories, alpha)
        current.append(NormalDistrubtion(Q_mu, Q_sigma,1, M)) #number of questions
        current.append(generateUserCompetencies(knowledgeGaps)) # knowledge gaps
        current.append(generateUserCompetencies(knowledgeGaps)) # prefrences currently not used
        users.append(current)
        
    return users

'''
Create a list of topics with different id. 
'''
def createTopics(numCategories):
    topics = []
    for i in range(numCategories):
        topics.append('topic'+str(i+1))
    return topics

'''
Create a list with questions, the number of questions is M. Each question has a question
id, a list of topic it covers, the difficulty(range 1-5) and the interest(range 1-5).
'''
def createQuestions(M,T, listOfTopics):
    questions = []
    for i in range(M):
        current = []
        current.append('q'+str(i)) #id
        current.append(getTopicfromDistribution(T,maxTopic,listOfTopics)) #topic
        current.append(NormalDistrubtion(D_mu, D_sigma,1,5)) #difficulty of question
        current.append(NormalDistrubtion(R_mu, R_sigma,1,5)) #rating of question
        questions.append(current)
    return questions
            
#createQuestions(5)  

'''
To calculate the student interest for the question using student's preference.
If the preference is greater than 0.5, the interest rating will be
above the average and vice versa.
The interest value must be between 1 to 5.
'''
def computeRating(mean, preference):
    return NormalDistrubtion(mean - 1 +2*preference, 0.5, MINRATING,MAXRATING)
# computeRating(2, 1)

'''
To calculate the difficulty for the question using knowledge gap for each student.
If the knowledge gap is greater than 0.5, the difficulty will be
above the average and vice versa.
The difficulty value must between 1 to 5.
'''
def computeDifficulty(mean, knowledgeGap):
    return NormalDistrubtion(mean- 1 +2*knowledgeGap, 0.5,MINRATING, MAXRATING)
# computeDifficulty(2, 1)

#def computeanswer(knowledgeGap):
#    return np.random.choice(np.arange(0,2), p=[knowledgeGap, 1-knowledgeGap])                
                

'''
Compute whether a student ansewers a question right or wrong. The probability is based on
the knowledge gap of the topic the question covers and the difficulty of the question.
Finally, randomly choose right(1) or wrong(-1) according to probability.
'''
def computeanswer(knowledgeGap, difficutly):
    competency = 1-knowledgeGap
    probofSuccess =  (1)/(1 + math.exp( -4 * (competency - difficutly)))
    #print competency, difficutly, probofSuccess
    return np.random.choice([-1,1], p=[1-probofSuccess, probofSuccess])     
  
    
'''
Calculate the interest and difficulty for each student and each question.
e.g. Student i, Question j. Find interest and difficulty for i on j.
To do this, go through all the topics that question j covers. Get average competency
of student i for these topics.
'''
def getAvgCompetencyAcrossTopics(competencies, questionTopics, listOfTopics):
    sumOfComp = 0
    for value in questionTopics:
        indexOfTopic = listOfTopics.index(value)
        sumOfComp= sumOfComp + competencies[indexOfTopic]
    return sumOfComp / float(len(questionTopics))
    
'''
Create output QT and SQ.
SQ: for each student, calculate the interst R and dificulty D using function above, then
use interst and dificulty to get the reuslt whether the student can do it correctly (A).
Then, put R D A into SQ table.
QT:for each topic the question covers, add topic tag into the table.
'''
def createOutput(N,M, Users, Questions, listOfTopics):
    selected = np.zeros((N, M))
    SQR = []
    SQD = []
    SQA= []
    QT = []
    SQ = []
    for u in range(N): # create  SQR, SQD, SQA
        numQ = Users[u][1] #number of questions done by user i
        for j in range(numQ):
            #pick new question to answer
            q = randint(0,M-1)
            while (selected[u][q]==1):
                q = randint(0,M-1)
            selected[u][q]=1
            R = computeRating(Questions[q][3], getAvgCompetencyAcrossTopics(Users[u][3],Questions[q][1],listOfTopics))
            #SQR.append([u, q, R])
            D = computeDifficulty(Questions[q][2],getAvgCompetencyAcrossTopics(Users[u][3],Questions[q][1],listOfTopics))
            #SQD.append([u, q, D])
            A = computeanswer(getAvgCompetencyAcrossTopics(Users[u][3],Questions[q][1],listOfTopics), (float(Questions[q][2])- MINRATING)/(MAXRATING-MINRATING))
            #SQA.append([u, q, A]) 
            SQ.append([Users[u][0], Questions[q][0], A, D, R])
    for q in range(M): # create  QT
        for topics in Questions[q][1]:
            #QT.append([Questions[q][0], Questions[q][1]])
            QT.append([Questions[q][0], topics])
    
    # return SQR, SQD, SQA, QT, SQ
    return SQ, QT

'''
Check if the current working directory has the file path.
Create one if does not exists.
'''
def checkCreate(inputfolder):
    directory = os.getcwd() + '/' + inputfolder
    if not os.path.exists(directory):
        os.makedirs(directory)
    
'''
Create a csv file using listname in the given file path. 
'''
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


'''
Create the whole dataset(Users.csv, Question.csv, QT.csv, SQ.csv).
To do this, generate the peak of competency firstly, use the peak competency
to generate users competency and create users. Also, use tags to generate a
list of topics the nuse the list to generate different questions. Then, use
users and questions to generate output QT and SQ. Finally, put users, questions,
QT,SQ into four tables.
'''
def createDataset(N, M, T, alpha): 
    checkCreate(inputfolder)
    listOfTopics = createTopics(numCategories)
    Users= createUsers(N, T)
    Questions = createQuestions(M,T,listOfTopics)
    SQ, QT = createOutput(N,M,Users,Questions,listOfTopics)
    createFile(inputfolder, 'Users.csv', Users, ['uid', 'numQuestions', 'Knowledge gaps', 'interests'])
    createFile(inputfolder, 'Questions.csv', Questions, ['qid', 'Topics', 'Avereage D', 'Average P'])
    createFile(inputfolder, 'QT.csv', QT, ['uid', 'qid'])
    createFile(inputfolder, 'SQ.csv', SQ, ['uid', 'qid', 'A', 'D', 'P'])    


# In[3]:

createDataset(N, M, numCategories, alpha)




