import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    st.title("Breast Cancer Wisconsin Dataset Explorer")

    # Veri seti açıklaması
    st.write("""
    Bu uygulama, Wisconsin Meme Kanseri Veri Seti'ni kullanarak meme kanseri teşhisi için farklı makine öğrenimi modellerini değerlendirir.
    """)

    # Veri setini yükleme
    st.sidebar.header('Veri Seti Yükleme')
    uploaded_file = st.sidebar.file_uploader("Lütfen bir CSV dosyası yükleyin", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success('Veri seti başarıyla yüklendi.')
    else:
        st.stop()

    # Veri ön işleme
    st.header('Veri Ön İşleme')
    st.subheader("Veri Seti İlk 10 Satırı:")
    st.write(data.head(10))

    data.drop("Unnamed: 32", axis=1, inplace=True)
    st.subheader("Korelasyon Matrisi:")
    corr_matrix = data.corr()
    st.write(corr_matrix)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Model seçimi
    st.sidebar.header('Model Seçimi')
    model_name = st.sidebar.selectbox("Lütfen bir model seçin", ['KNN', 'SVM', 'Naive Bayes'])

    if model_name == 'KNN':
        st.sidebar.info('K-En Yakın Komşu (KNN) sınıflandırıcısı, veri noktalarının etrafındaki k-nearest (k-en yakın) komşuların sınıf etiketlerine dayanarak sınıflandırma yapar.')
    elif model_name == 'SVM':
        st.sidebar.info('Destek Vektör Makineleri (SVM), veri noktalarını sınıflandırmak için bir hiper düzlem oluşturur ve bu düzlemin iki tarafındaki sınıfları ayırmaya çalışır.')
    elif model_name == 'Naive Bayes':
        st.sidebar.info('Naive Bayes sınıflandırıcısı, Bayes Teoremi ve özellikler arasındaki bağımsızlık varsayımına dayanarak sınıflandırma yapar.')

    # Veriyi ayırma
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model eğitimi ve sonuçları
    st.header('Model Sonuçları')
    if model_name == 'KNN':
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
    elif model_name == 'SVM':
        model = SVC()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
        param_grid = {}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    st.subheader('En İyi Parametreler:')
    st.write(best_params)

    # Model sonuçları
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    st.write("Confusion Matrix:")
    st.write(confusion)

if __name__ == "__main__":
    main()
