import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Veri setini yükleme
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Data ön işlemleri
def preprocess_data(data):
    # Gereksiz sütunları temizle
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    
    # M ve B değerlerini 1 ve 0 olarak değiştirme
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Veriyi X ve Y olarak ayırma
    X = data.drop('diagnosis', axis=1)
    Y = data['diagnosis']
    
    # Veriyi train ve test setlerine ayırma
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

# Model implementasyonu
def train_model(X_train, Y_train, model_name):
    if model_name == 'KNN':
        model = KNeighborsClassifier()
        params = {'n_neighbors': [3, 5, 7, 9]}
    elif model_name == 'SVM':
        model = SVC()
        params = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
        params = {}
    
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    
    best_params = grid_search.best_params_
    st.write('En iyi parametreler:', best_params)
    
    model = grid_search.best_estimator_
    return model

# Model analizi
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    confusion = confusion_matrix(Y_test, Y_pred)
    
    st.write('Accuracy:', accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1 Score:', f1)
    st.write('Confusion Matrix:', confusion)

def main():
    st.title('Breast Cancer Wisconsin Data Set Analizi')
    st.sidebar.title('Veri Seti Seçimi')
    
    # Veri setini seçme
    file_path = st.sidebar.file_uploader('Veri setini yükleyin', type='csv')
    
    if file_path is not None:
        data = load_data(file_path)
        X_train, X_test, Y_train, Y_test = preprocess_data(data)
        
        st.subheader('İlk 10 Satır:')
        st.write(data.head(10))
        
        st.subheader('Sütunlar:')
        st.write(data.columns)
        
        model_name = st.sidebar.selectbox('Model Seçimi', ('KNN', 'SVM', 'Naive Bayes'))
        
        if st.sidebar.button('Modeli Eğit'):
            st.write('Model eğitiliyor...')
            model = train_model(X_train, Y_train, model_name)
            st.write('Model eğitildi.')
            
            st.subheader('Model Analizi')
            evaluate_model(model, X_test, Y_test)

if __name__ == '__main__':
    main()

