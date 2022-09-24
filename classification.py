
import streamlit as st
import pandas as pd
from sklearn import datasets,tree
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image as img


image = img.open("iris-machinelearning.png")

st.write("""# IRIS CLASSIFICATION

This app compares different classification alogorithms for Iris Dataset""")

st.image(image)

st.subheader("Dataset Description")
st.write("""The Iris Dataset contains four features
(length and width of sepals and petals)
of 50 samples of three species of Iris
(Iris setosa, Iris virginica and Iris versicolor).
These measures were used to create a linear discriminant model
to classify the species. The dataset is often used in data mining,
classification and clustering examples and to test algorithms.""")

iris = datasets.load_iris()

attr = ['Sepal length','Sepal width','Petal length','Petal width']
classes = ['Iris-Setosa','Iris-Versicolor','Iris-Virginica']


sepal_length = st.sidebar.slider(attr[0],4.3,7.9,5.4)
sepal_width = st.sidebar.slider(attr[1],2.0,4.4,3.6)
petal_length = st.sidebar.slider(attr[2],1.0,6.9,1.3)
petal_width = st.sidebar.slider(attr[3],0.1,2.5,0.2)


data = np.array([sepal_length,
sepal_width ,
petal_length,
petal_width ]).reshape(1,-1)

st.subheader('User Input')
st.write(data)

size = st.sidebar.slider('Test Size',0,100,25)

st.write('Test Size: ',size,"%")

if size < 3:
    st.error('Cannot be less than 3')

X,Y = iris.data, iris.target

st.subheader('Attributes')
st.write(np.array(attr).reshape(1,4))
st.subheader('Classes')
st.write(np.array(classes).reshape(1,3))

train_x,test_x,train_y,test_y = tts(X,Y,test_size=size,random_state=10)

def metrics(ydash):
    return accuracy_score(test_y,ydash)

def naive_bayes(params):
    
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(train_x,train_y)
    ydash = classifier.predict(test_x)
    classified  = classifier.predict(params)

    return ydash,classified

def k_nearest(params):

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    classifier.fit(train_x,train_y)
    ydash = classifier.predict(test_x)
    classified = classifier.predict(params)

    return ydash,classified


def decision_tree(params):

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=10,random_state=5,max_features=None
                                  ,min_samples_leaf=15)
    tree.fit(train_x,train_y)
    ydash = tree.predict(test_x)
    classified = tree.predict(params)

    return ydash, classified

def random_forest(params):

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(n_estimators=70,random_state=10,min_samples_leaf=25)
    classifier.fit(train_x,train_y)
    ydash = classifier.predict(test_x)
    classified = classifier.predict(params)

    return ydash, classified

st.write('# PREDICTION ANALYSIS')


ydashnb, nbpred = naive_bayes(data)
ydashknn, knnpred = k_nearest(data)
ydashds, dspred = decision_tree(data)
ydashrf, rfpred = random_forest(data)


final = pd.DataFrame({
    'Classifier':['Naive-Bayes','K-Nearest Neighbor',
                 'Decision Tree','Random Forest'],
                 
    'Predicted Output':[classes[int(nbpred)],classes[int(knnpred)],
    classes[int(dspred)],classes[int(rfpred)]],

    'Model Accuracy':[metrics(ydashnb),metrics(ydashknn),metrics(ydashds),metrics(ydashrf)]
}, index = range(1,5))

st.write(final)


