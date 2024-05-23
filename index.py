# Mengimpor library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from joblib import load,dump
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time

def train():
    df = pd.read_csv('Diabetes.csv')

    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df >= lower_bound) & (df <= upper_bound)].dropna()


    numerical_columns = df.select_dtypes(include=['number']).columns

    for col in numerical_columns:
        df[col] = remove_outliers(df[col])

    df = df.dropna()



    scaling_factor = 100


    scaled = (df['DiabetesPedigreeFunction'] * scaling_factor)


    df.loc[:, 'DiabetesPedigreeFunction'] = scaled

    #
    scaler = StandardScaler()

    # Memilih fitur yang akan digunakan untuk melatih model
    fitur = ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Umur']

    # Membuat DataFrame untuk data train (Ketebalan Kulit tidak null dan bukan nol)
    data_train = df[(df['Ketebalan Kulit'].notna()) & (df['Ketebalan Kulit'] != 0)]

    # Membuat DataFrame untuk data test (Ketebalan Kulit null atau nol)
    data_test = df[(df['Ketebalan Kulit'].isna()) | (df['Ketebalan Kulit'] == 0)]


    # Memisahkan fitur dan target
    X_train = data_train[fitur]
    y_train = data_train['Ketebalan Kulit']
    X_test = data_test[fitur]

    # Scale fitur


    # List Algoritma Regresi
    models = [
        RandomForestRegressor(max_features=5),
        GradientBoostingRegressor(),
        SVR(kernel="poly",degree=1),
        KNeighborsRegressor()
    ]


    cv_scores = {}


    for model in models:

        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

        mean_mae = -scores.mean()

        cv_scores[model.__class__.__name__] = mean_mae

    sorted_cv_scores = {k: v for k, v in sorted(cv_scores.items(), key=lambda item: item[1])}



    # Memilih MAE terendah
    best_model_name = list(sorted_cv_scores.keys())[0]
    best_model = [model for model in models if model.__class__.__name__ == best_model_name][0]
    # Train menggunakan skor terendah ( terbaik )
    best_model.fit(X_train, y_train)


    predicted_ketebalan_kulit = best_model.predict(X_test)




    predicted_ketebalan_kulit = predicted_ketebalan_kulit.round().astype(int)


    df.loc[(df['Ketebalan Kulit'] == 0.0), 'Ketebalan Kulit'] = predicted_ketebalan_kulit


    # Mengganti nilai 0 pada kolom Insulin dengan prediksi machnine learning terbaik

    # Memilih fitur yang akan digunakan untuk melatih model
    fitur = ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit', 'BMI', 'DiabetesPedigreeFunction','Umur']

    # Membuat DataFrame untuk data train (Insulin tidak null dan bukan nol)
    data_train = df[(df['Insulin'].notna()) & (df['Insulin'] != 0)]

    # Membuat DataFrame untuk data test (Insulin null atau nol)
    data_test = df[(df['Insulin'].isna()) | (df['Insulin'] == 0)]

    # Memisahkan fitur dan target
    X_train = data_train[fitur]
    y_train = data_train['Insulin']
    X_test = data_test[fitur]

    # Modeling
    models = [
        RandomForestRegressor(max_features=5),
        GradientBoostingRegressor(),
        SVR(kernel="linear"),
        KNeighborsRegressor()
    ]

    cv_scores = {}

    for model in models:
        
            scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        
            mean_mae = -scores.mean()
        
            cv_scores[model.__class__.__name__] = mean_mae

    sorted_cv_scores = {k: v for k, v in sorted(cv_scores.items(), key=lambda item: item[1])}

    # Memilih MAE terendah
    best_model_name = list(sorted_cv_scores.keys())[0]
    best_model = [model for model in models if model.__class__.__name__ == best_model_name][0]
    # Train menggunakan skor terendah ( terbaik )
    best_model.fit(X_train, y_train)

    predicted_insulin = best_model.predict(X_test)

    predicted_insulin = predicted_insulin.round().astype(int)

    df.loc[(df['Insulin'] == 0.0), 'Insulin'] = predicted_insulin

    # Drop kolom yang tidak diperlukan dari DataFrame
    features = df.drop(columns=['Hasil'])

    # Pisahkan fitur dan label
    X = features
    y = df['Hasil']

    # Bagi data menjadi set pelatihan dan set pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi objek StandardScaler
    scaler = StandardScaler()

    # Fit scaler ke data pelatihan dan transformasikan data pelatihan
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformasikan data pengujian menggunakan scaler yang sama
    X_test_scaled = scaler.transform(X_test)

    # Inisialisasi model SVM
    model = SVC(kernel='poly',degree=1,random_state=42)

    # Latih model dengan set pelatihan yang telah di-scale
    model.fit(X_train_scaled, y_train)

    # Lakukan prediksi pada set pengujian yang telah di-scale
    predictions = model.predict(X_test_scaled)

    # Hitung akurasi model
    accuracy = accuracy_score(y_test, predictions)
    #save model, accuracy, and scaler into one object then save it to disk
    trained_model = {'model':model,'accuracy':accuracy,'scaler':scaler}
    dump(trained_model,'trained_model.joblib')
    return accuracy


is_trained = False
#try to load the model from disk if it exist then set is_trained to True
try:
    trained_model = load('trained_model.joblib')
    is_trained = True
except:
    is_trained = False



if(is_trained):
    trained_model = load('trained_model.joblib')
    model = trained_model['model']
    accuracy = trained_model['accuracy']
    scaler = trained_model['scaler']



    st.title('Pendeteksi Penyakit Diabetes Menggunakan Algoritma Support Vector Machine')
    with st.container(border=True):
        st.markdown(f'''
                    Aplikasi ini mampu memprediksi apakah seseorang menderita penyakit diabetes atau tidak berdasarkan data yang diinputkan dengan tingkat akurasi sebesar  :green[{accuracy*100}%]
                    ''')
        if 'button_status' not in st.session_state:
            st.session_state.button_status = 'button_active'

        # Button to start retraining the model
        if(st.session_state.button_status == 'button_active'):
            if st.button('Re-Train Model'):
                st.session_state.button_status = 'training'
                st.rerun()
        elif(st.session_state.button_status == 'training'):
            with st.spinner('Proses melatih model...'):
                accuracy = train()
            st.session_state.retrain_accuracy = accuracy
            st.session_state.button_status = 'button_active'
            st.rerun()

        
        if('retrain_accuracy' in st.session_state):
            st.markdown('''Model behasil dilatih dengan akurasi sebesar :green[{}]'''.format(st.session_state.retrain_accuracy*100))
            #remove retrain_accuracy from session state
            del st.session_state['retrain_accuracy']
            time.sleep(5)
            st.rerun()
    
    # Display UI elements immediately
    st.markdown('''## Input Data Pasien''')
    st.write('Silahkan masukkan data pasien yang ingin diprediksi pada kolom di bawah ini')




    #Kehamilan,Glukosa,Tekanan Darah,Ketebalan Kulit,Insulin,BMI,DiabetesPedigreeFunction,Umur input
    kehamilan = st.number_input('Kehamilan', min_value=0, max_value=17, value=0)
    glukosa = st.number_input('Glukosa', min_value=0, max_value=200, value=0)
    tekanan_darah = st.number_input('Tekanan Darah', min_value=0, max_value=200, value=0)
    ketebalan_kulit = st.number_input('Ketebalan Kulit', min_value=0, max_value=200, value=0)
    insulin = st.number_input('Insulin', min_value=0, value=0)
    bmi = st.number_input('BMI', min_value=0.0, value=0.0)
    dpf = st.number_input('DiabetesPedigreeFunction', min_value=0.000, max_value=3.000, value=0.000, format="%.3f")
    umur = st.number_input('Umur', min_value=21, max_value=100, value=21)





    #button "Prediksi"
    if st.button('Prediksi'):
        # Membuat DataFrame dari data pasien
        data = {'Kehamilan': [kehamilan],
            'Glukosa': [glukosa],
            'Tekanan Darah': [tekanan_darah],
            'Ketebalan Kulit': [ketebalan_kulit],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Umur': [umur]}
        df_input = pd.DataFrame(data)
        # Transformasi data pasien menggunakan StandardScaler yang sama
        df_input_scaled = scaler.transform(df_input)

        # Prediksi apakah pasien menderita diabetes atau tidak warna merah = menderita diabetes, warna hijau = tidak menderita diabetes pada tulisan hasil prediksi
        prediction = model.predict(df_input_scaled)

        if prediction[0] == 1:
            st.markdown('''
                        Pasien :red[MENDERITA] diabetes
                        ''')
        else:
            st.markdown('''
                        Pasien :green[TIDAK MENDERITA] diabetes
                        ''')



else:

    st.title('Pendeteksi Penyakit Diabetes Menggunakan Algoritma Support Vector Machine')
    st.write('Belum ada model yang dilatih')
    
    if 'button_status' not in st.session_state:
        st.session_state.button_status = 'button_active'

    # Button to start retraining the model
    if(st.session_state.button_status == 'button_active'):
        if st.button('Train Model'):
            st.session_state.button_status = 'training'
            st.rerun()
    elif(st.session_state.button_status == 'training'):
        with st.spinner('Proses melatih model...'):
            accuracy = train()
        st.session_state.retrain_accuracy = accuracy
        st.session_state.button_status = 'no_button'
        st.markdown('''Model behasil dilatih dengan akurasi sebesar :green[{}]%'''.format(accuracy*100))
        time.sleep(5)
        st.rerun()

