import numpy as np
import pandas as pd

data = pd.read_csv('userResponse.csv')
data = data.drop('Timestamp', axis=1)

dnumFavSub = 13
dnumPersonalTraits = 12
dnumSubjectScores = 5
dnumTotalOptions = dnumFavSub + dnumPersonalTraits + 4 + dnumSubjectScores

fullSubjects = ['Physics', 'Chemistry', 'Mathematics', 'Biology', 'Computer Science', 'Business Studies', 'Accountancy', 'Economics', 'Literature', 'Psychology', 'History', 'Geography', 'Political Science (Civics)']
fullTraits = ['Extrovert', 'Introvert', 'Attention Loving', 'Focused', 'Creative', 'Patient', 'Proactive', 'Leader', 'Hardworking', 'Calculative', 'Good @ Communication', 'Good Business Accumen']
fullBoard = ['CBSE' , 'ICSE' , 'State Board' , 'IB']

def fillFavSub(inputVector, strSubjects):
    
    for i in range(0,dnumFavSub):
        if fullSubjects[i] in strSubjects:
            inputVector[i] = 1

    return inputVector  

def fillAcads(inputVector,AcadLiking):

    inputVector[dnumFavSub] = AcadLiking     # dnumFavSub becasue it is filled post that
    
    return inputVector

def fillpersTraits(inputVector, strTraits):
    
    for i in range(0,dnumPersonalTraits):
        if fullTraits[i] in strTraits:
            inputVector[dnumFavSub + 1 + i] = 1

    return inputVector

def fillScopeInterest(inputVector, option):
    
    if(option == "Generalist (Like to know little about a lot of things)"):
        inputVector[dnumFavSub + 1 + dnumPersonalTraits] = 1
    
    return inputVector

def fillMonetary(inputVector,score):
    inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1] = score
    return inputVector

def fillBoard(inputVector,Board):
    for i in range(4):
        if(fullBoard[i] == Board):
            inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1 + 1] = i
            break
    
    return inputVector

def fillMarks(inputVector,currDataSeries):
    
    inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1 + 1 + 1] = currDataSeries['MATHEMATICS (/100)']
    inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1 + 1 + 2] = currDataSeries['SCIENCE (/100)']
    inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1 + 1 + 3] = currDataSeries['ENGLISH (/100)']
    inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1 + 1 + 4] = currDataSeries['HINDI (/100)']
    inputVector[dnumFavSub + 1 + dnumPersonalTraits + 1 + 1 + 5] = currDataSeries['SOCIAL STUDIES (/100)']

    return inputVector
    
def fillVector(inputVector,currDataSeries):
    inputVector = fillFavSub(inputVector,currDataSeries['What are your favorite subjects?'])
    inputVector = fillAcads(inputVector,currDataSeries['How much do you like Academics'])
    inputVector = fillpersTraits(inputVector,currDataSeries['What suits your personality'])
    inputVector = fillScopeInterest(inputVector,currDataSeries['How is your scope of interest'])
    inputVector = fillMonetary(inputVector,currDataSeries['Choose from scale how inclined you are'])
    inputVector = fillBoard(inputVector,currDataSeries['Select your Board'])
    inputVector = fillMarks(inputVector,currDataSeries)

    return inputVector




def transformToVector(dataFrame):
    vector_list = []
           
    for index in range(dataFrame.shape[0]):
        inputVector = [0]*dnumTotalOptions
        outputVector = fillVector(inputVector , dataFrame.iloc[index])
        packagedVector = [dataFrame.iloc[index]['email-Ids'] , outputVector]
        vector_list.append(packagedVector)

    return vector_list

List_of_vectors = transformToVector(data)

print(List_of_vectors)