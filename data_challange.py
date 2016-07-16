#############################################################################################################

import csv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

##############################################################################################################
#...............................Loading Training Dataset from tsv file.......................................#
##############################################################################################################


tracker_data = pd.read_csv("classification_train.tsv", sep='\t',header = None, low_memory = False)

#############################################################################################################
#........................Labeling the column names as there is no header in input file......................#
#############################################################################################################


tracker_data = tracker_data.rename(columns = {0:'Product_title',1:'Product_ID',2:'Brand_ID'})


#############################################################################################################
#................................Loading Test Dataset from tsv file.........................................#
#############################################################################################################


tracker_data_test = pd.read_csv("classification_blind_set_corrected.tsv", sep='\t',header = None, low_memory = False)


#############################################################################################################
#.................Labeling the column names as there is no header in test dataset...........................#
#############################################################################################################


tracker_data_test = tracker_data_test.rename(columns = {0:'Product_title',1:'Brand_ID'})


##############################################################################################################
#.............Creating a dictionary to store all the product titile from the training........................# 
#.............dataset according to each brand id. Here key values are brand id and...........................#
#.......................product titles are the corressponding values.........................................#
##############################################################################################################


p=np.empty(0,int)
l={k: list(v) for k,v in tracker_data.groupby("Brand_ID")["Product_title"]}
p=l.keys()


##############################################################################################################
#................Creating a dictionary to store all the product titile from..................................#
#................the training dataset according to each brand id. Here key...................................#
#.............values are brand id and product titles are the corressponding values...........................#
##############################################################################################################


p2=np.empty(0,int)
l2={k: list(v) for k,v in tracker_data.groupby("Product_ID")["Product_title"]}
p2=l.keys()


##############################################################################################################
#..................Finding the total number of brand id in the training dataset..............................#
##############################################################################################################

c=len(p)


##############################################################################################################
#...................finding the top two heighest number of occourance of words in............................#
#...................each brand Id. Because based on this words brand id is selected..........................#
##############################################################################################################


b=[]
for i in range(0,c):
    a=l.values()[i]
    from collections import Counter
    import re
    counts = Counter()
    words = re.compile(r'\w+')

    for sentence in a:
            counts.update(words.findall(sentence.lower()))

    for k, v in counts.most_common(2):
        b.append(k)
wrd=np.array(b)
del b


############################################################################################################
#......delete those maximum words from product title and selecting all the product id into an array........#
############################################################################################################


h=[]
m=[]
y=len(tracker_data)

for i in range(0,y):
    sentence1 = tracker_data['Product_title'].iloc[i]
    cat =(tracker_data['Product_ID'].iloc[i])
    cat=str(cat)
    bom=[]
    bom.append(cat)
    m.append(bom)
    word_list = sentence1.split()
    word_list = map(str.lower,word_list)

    fin=' '.join([q for q in word_list if q not in wrd])
    h.append(fin)

mnp=np.array(m)


#####################################################################################################
#.................Removing those maximum word from test data set product title......................#
#####################################################################################################


h_test=[]

y_test=len(tracker_data_test)

for i in range(0,y_test):
    sentence1_test = tracker_data_test['Product_title'].iloc[i]
    word_list_test = sentence1_test.split()
    word_list_test = map(str.lower,word_list_test)

    fin_test=' '.join([q for q in word_list_test if q not in wrd])

    h_test.append(fin_test)
h_testnp=np.array(h_test)
del h_test


#####################################################################################################
#....................Applying Machine Learning technique to train the test data.....................#
#.........As there are many categorical product Id, so normal SVM will not give.....................#
#.........the proper output. For multilabel categorical product Id SVM One vs Rest..................#
#.........................Classifier will give the maximum possible output..........................#
#####################################################################################################


lb = preprocessing.LabelBinarizer() 

Y = lb.fit_transform(m) 

classifier =Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC()))]) 


######################################################################################################
#......................................Model from train data.........................................#
######################################################################################################


classifier.fit(h, Y) 


######################################################################################################
#......................................Training the test data........................................#
######################################################################################################


predicted = classifier.predict(h_testnp) 
all_labels = lb.inverse_transform(predicted)


######################################################################################################
#..............Printing the final output with the product id for each product title..................#
######################################################################################################


with open('mydata.csv', 'w') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile)
    for labels in all_labels:
        thedatawriter.writerow(labels)

############################################################################################################

