# -*- coding: utf-8 -*-

# -- Sheet --

import re
from collections import Counter

#Create Counters for both English and Swedish words
Ewords = re.findall(r'\w+', open('europarl-v7.sv-en.lc.en').read())
Swords = re.findall(r'\w+', open('europarl-v7.sv-en.lc.sv').read())
Ecnt = Counter(Ewords)
Scnt = Counter(Swords)

#Print the 10 most frequent words in each language
print(Ecnt.most_common(10))
print(Scnt.most_common(10))

#Print probability in percent of the words 'speaker' and 'zebra'
print(Ecnt['speaker']/sum(Ecnt.values())*100)
print(Ecnt['zebra']/sum(Ecnt.values())*100)

#Function to find all bigrams in a language file
def createBigram(languageFile):
   bigrams = []
   countBigrams = {}
   countUnigram = {}
   
   #Create list of bigrams and count the number of the occurance of each bigram and unigram
   for i in range(len(languageFile)-1):
      if i < len(languageFile) - 1 and languageFile[i+1]:
         bigrams.append((languageFile[i], languageFile[i + 1]))       
         if (languageFile[i], languageFile[i+1]) in countBigrams:
            countBigrams[(languageFile[i], languageFile[i + 1])] += 1     
         else:
            countBigrams[(languageFile[i], languageFile[i + 1])] = 1
      if languageFile[i] in countUnigram:
         countUnigram[languageFile[i]] += 1
      else:
         countUnigram[languageFile[i]] = 1
   return bigrams, countUnigram, countBigrams


#Function to determine the sentence probability from a given language file
def findProbability(sentence, languageFile):
    splt=sentence.split()
    probability = 1
    bilist = []
    bigrams, countUnigram, countBigrams = createBigram(languageFile)
        
    #Calculate the probability of the bigrams in the language file
    listOfProb = {}
    for bigram in bigrams:
        word1 = bigram[0]
        word2 = bigram[1]
        listOfProb[bigram] = (countBigrams.get(bigram)+1)/(countUnigram.get(word1)+len(countUnigram))
         
    #Create bigrams for the input sentence    
    for i in range(len(splt) - 1):
        if i < len(splt) - 1:
            bilist.append((splt[i], splt[i+1]))
        
    #Calculate the probability of the input sentence
    for i in range(len(bilist)):
        if bilist[i] in listOfProb:
            probability *= listOfProb[bilist[i]]
        elif countUnigram.get(bilist[i][0]) != None:
            probability *= 1/(countUnigram.get(bilist[i][0])+ len(countUnigram))
        else:
            probability *= 1/(len(countUnigram))
    print('The probablility of "' + sentence + '": ' + str(probability))
    return probability

sentence = "i declare that"
findProbability(sentence,Ewords)

#Create arrays with all sentences from language file
fpE = open('europarl-v7.sv-en.lc.en')
dataE = fpE.read()
listofSentencesE = dataE.splitlines()

fpS = open('europarl-v7.sv-en.lc.sv')
dataS = fpS.read()
listofSentencesS = dataS.splitlines()

#Create sentence pairs
listPairs = []
count = 0
for line in listofSentencesE:
    listPairs.append([listofSentencesS[count],listofSentencesE[count]])
    count = count + 1

sentencePairs = []
for line in listPairs:
    sentencePairs.append([line[0].split(),line[1].split()])

#Remove dots and commas from sentencePairs
for pair in sentencePairs:
    while '.' in pair[0]:
        pair[0].remove('.')
    while '.' in pair[1]:
        pair[1].remove('.')
    while ',' in pair[0]:
        pair[0].remove(',')
    while ',' in pair[1]:
        pair[1].remove(',')

#Function to initialize probabilities to 1/(length of Swedish vocabulary)
def initialize(t, initial_value, sentencePairs):
    count = 0
    for pairs in sentencePairs:
        for sword in sentencePairs[count][0]:
            for eword in sentencePairs[count][1]:
                tup = (eword, sword)
                t[tup] = initial_value
        count = count + 1

#Initialize probabilities and create dictionaries
t = {}
initial_value = 1.0 / len(Scnt)
initialize(t, initial_value, sentencePairs)
s_total = {} 

# Loop through a number of EM iterations
for iter in range(0,10):
    
    #Initialize and set counts to 0
    bigramCount = {} 
    unigramCount = {} 
    cnt = 0
    for pairs in sentencePairs:
        for sword in sentencePairs[cnt][0]:
            unigramCount[sword] = 0
            for eword in sentencePairs[cnt][1]:
                bigramCount[(eword,sword)] = 0
        cnt = cnt + 1
        
    #Expectaion cycle
    for pairs in sentencePairs:
        for eword in pairs[1]:
            s_total[eword] = 0
            for sword in pairs[0]:
                s_total[eword] += t[(eword, sword)]

        #Update counts
        for eword in pairs[1]:
            for sword in pairs[0]:
                bigramCount[(eword, sword)] += t[(eword, sword)]/s_total[eword]
                unigramCount[sword] += t[(eword, sword)]/s_total[eword]

    #Maximization cycle. Estimate probabilities
    cnt = 0
    for pairs in sentencePairs:
        for sword in sentencePairs[cnt][0]:
            for eword in sentencePairs[cnt][1]:
                t[(eword,sword)] = bigramCount[(eword,sword)]/unigramCount[sword]
        cnt = cnt + 1
    
    #For each iteration, print 10 most likely Swedish translations for the English word european
    translation = []
    for pairs in t.keys():
        if pairs[0] == "european":
            translation.append([pairs[1],t[("european",pairs[1])]])
            translation = sorted(translation, key=lambda x: x[1], reverse=True)
    print(translation[0:10])

#Function to translate English word to Swedish
def translateWord (eword):
    translation = []
    for pairs in t.keys():
        if pairs[0] == eword:
            translation.append([pairs[1],t[(eword,pairs[1])]])
            translation = sorted(translation, key=lambda x: x[1], reverse=True)
    return translation[0][0]

translateWord('declare')

#Function to translate English sentence to Swedish
def translateSent (eSentence):
    words = eSentence.split()
    sSentence = ""
    for word in words: 
        sSentence = sSentence + " " + translateWord(word)
    return sSentence

trans = translateSent('i declare')
print(trans)

def translateWord2 (eword):
    translation = []

    for pairs in t.keys():
        if pairs[0] == eword:
            translation.append([pairs[1],t[(eword,pairs[1])]])
            translation = sorted(translation, key=lambda x: x[1], reverse=True)

    return translation[0:15]

def translateSent2 (eSentence):
    
    words = eSentence.split()
    possibleTranslations = []
    p = {}
    sSentence = []
    sSentence.append(translateWord2(words[0])[0][0])
    
    for word in words:
        possibleTranslations.append(translateWord2(word))
    
    count = 0

    while count < len(possibleTranslations) - 1:
        probabilities = []
        
        for i in range (0,10):
            word = possibleTranslations[count+1][i][0]
            prob = findProbability(sSentence[count] + " " + word, Swords)
            probabilities.append([word,prob])

        maxProb = 0
        for prob in probabilities:
            if prob[1] > maxProb:
                maxProb = prob[1]
    
        for prob in probabilities:
            if prob[1] == maxProb:
                sSentence.append(prob[0])
                break
        count = count + 1
    return sSentence

trans = translateSent2('i declare resumed the session of the european parliament adjourned')
print(trans)

