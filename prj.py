#import csv
#main_data = []
#with open("classification_train.tsv") as tsvfile:
#    tsvreader = csv.reader(tsvfile, delimiter="\t")
#    for line in tsvreader:
#        main_data.append(line)
#        
#print(main_data[1:])
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer
tracker_data = pd.read_csv("classification_train.tsv", sep='\t',header = None, low_memory = False)
tracker_data = tracker_data.rename(columns = {0:'Product_title',1:'Product_ID',2:'Brand_ID'})
print(type(tracker_data))
import numpy as np
import pandas as pd
tracker_data_test = pd.read_csv("classification_blind_set_corrected.tsv", sep='\t',header = None, low_memory = False)
tracker_data_test = tracker_data_test.rename(columns = {0:'Product_title',1:'Brand_ID'})

#print(tracker_data.head(1))
#cluster_8=tracker_data.query('Brand_ID == "8"')
#print(cluster_8)
#factor_3 = tracker_data['Brand_ID']
#counts = factor_3.value_counts()
#print(counts[0])
#x=[]
#for i in range (len(counts)):
#    if counts[i] > 50:
#        x.append(counts.index[i])
#print(x[1])
l={k: list(v) for k,v in tracker_data.groupby("Brand_ID")["Product_title"]}
p=l.keys()

l2={k: list(v) for k,v in tracker_data.groupby("Product_ID")["Product_title"]}
p2=l.keys()

c=len(p)
z=c+1
#print(l.items()[0])
b=[]
for i in range(0,c):
#    print(l.values()[i])

    a=l.values()[i]
#print(a)
    from collections import Counter
    import re

    counts = Counter()
    words = re.compile(r'\w+')
#    words = words.lower()

    for sentence in a:
        counts.update(words.findall(sentence.lower()))

#c=counts.most_common()

    for k, v in counts.most_common(2):
        b.append(k)
#print(b)
h=[]
m=[]
y=len(tracker_data)

for i in range(0,y/100):
    sentence1 = tracker_data['Product_title'].iloc[i]
    cat =(tracker_data['Product_ID'].iloc[i])
    cat=str(cat)
    b=[]
    b.append(cat)
    m.append(b)
#    print(sentence1)
    remove_list = b
    word_list = sentence1.split()
    word_list = map(str.lower,word_list)
#    print(word_list)
    fin=' '.join([q for q in word_list if q not in remove_list])
#    print(fin)
    h.append(fin)

h_test=[]

y_test=len(tracker_data_test)

for i in range(0,y_test/100):
    sentence1_test = tracker_data_test['Product_title'].iloc[i]
    
#    print(sentence1)
    remove_list = b
    word_list_test = sentence1_test.split()
    word_list_test = map(str.lower,word_list_test)
#    print(word_list)
    fin_test=' '.join([q for q in word_list_test if q not in remove_list])
#    print(fin)
    h_test.append(fin_test)
print(len(h_test))
#print(h[0:5])
#m=[]
#for i in range(0,y):
#    cat = tracker_data['Product_ID'].iloc[i]
#    m.append(cat)

#from sklearn.feature_extraction.text import TfidfVectorizer
#vect_test = TfidfVectorizer(h_test)
#F_test = vect_test.fit_transform(h_test)
#print(len(h))
#print(h[0:5])
#m=[]
#for i in range(0,y):
#    cat = tracker_data['Product_ID'].iloc[i]
#    m.append(cat)

#from sklearn.feature_extraction.text import TfidfVectorizer
#vect = TfidfVectorizer(h)
#F = vect.fit_transform(h)
#print(F)
#Create a Gaussian Classifier
#model = MultinomialNB()
#print(model)
# Train the model using the training sets 
#model.fit(F,m)
#print(len(m))

#print(c[0])
###############################################################################








#print(len(F_test))
#final_model = model.predict(F_test)
#print(len(final_model))
#print(y_test)


#print(c[0])


#######################################################################################
X_train = np.array(h) 

y_train_text = m

X_test = np.array(h_test) 

target_names = p2

lb = preprocessing.LabelBinarizer() 

Y = lb.fit_transform(y_train_text) 

classifier =Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC()))]) 

classifier.fit(X_train, Y) 
predicted = classifier.predict(X_test) 
all_labels = lb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print(item,', '.join(labels))

print(m[1:4])
print(tracker_data['Product_ID'].iloc[1])
