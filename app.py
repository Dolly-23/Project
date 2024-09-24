import mysql
from flask import Flask,render_template,url_for,request
from mysql.connector import cursor
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd

mydb = mysql.connector.connect(host='localhost',user='root',password="",port='3306',database='anemia')
app=Flask(__name__)

def preprocessing(file):
    file.drop(index=file.index[0], axis=0, inplace=True)
    file.rename(columns={'Sex  ':'Sex','Age      ': 'Age', '  RBC    ': 'RBC', 'MCV  ': 'MCV', ' MCHC  ': 'MCHC', ' RDW    ': 'RDW',
                       ' PLT /mm3': 'PLT', ' HGB ': 'HGB'}, inplace=True)
    return file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registration',methods=['POST','GET'])
def registration():
    if request.method=="POST":
        print('a')
        un=request.form['name']
        print(un)
        em=request.form['email']
        pw=request.form['password']
        print(pw)
        cpw=request.form['cpassword']
        if pw==cpw:
            sql = "SELECT * FROM hmg"
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails=cur.fetchall()
            mydb.commit()
            all_emails=[i[2] for i in all_emails]
            if em in all_emails:
                return render_template('registration.html',msg='a')
            else:
                sql="INSERT INTO hmg(name,email,password) values(%s,%s,%s)"
                values=(un,em,pw)
                cursor=mydb.cursor()
                cur.execute(sql,values)
                mydb.commit()
                cur.close()
                return render_template('registration.html',msg='success')
        else:
            return render_template('registration.html',msg='Enter The Correct Password')
    return render_template('registration.html')

@app.route('/login',methods=["POST","GET"])
def login():
    if request.method=="POST":
        em=request.form['email']
        print(em)
        pw=request.form['password']
        print(pw)
        cursor=mydb.cursor()
        sql = "SELECT * FROM hmg WHERE email=%s and password=%s"
        val=(em,pw)
        cursor.execute(sql,val)
        results=cursor.fetchall()
        mydb.commit()
        print(results)
        print(len(results))
        if len(results) >= 1:
            return render_template('andb.html',msg='login succesful')
        else:
            return render_template('login.html',msg='Invalid Credentias')


    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/andb')
def diabetes():
    return render_template('andb.html')
@app.route('/homediabetes')
def homediabetes():
    return render_template('homediabetes.html')
# @app.route('/homealz')
# def homealz():
#     return render_template('homealz.html')


@app.route('/upload',methods=['POST','GET'])
def upload():
    global df
    if request.method=="POST":
        file=request.files['file']
        print(type(file.filename))
        print('hi')
        df=pd.read_csv(file)
        print(df.head(2))
        return render_template('upload.html', msg='Dataset Uploaded Successfully')
    return render_template('upload.html')

@app.route('/view_data')
def view_data():
    print(df)
    print(df.head(2))
    print(df.columns)
    return render_template('viewdata.html',columns=df.columns.values,rows=df.values.tolist())
@app.route('/split',methods=["POST","GET"])
def split():
    global X,y,X_train,X_test,y_train,y_test
    if request.method=="POST":
        size=int(request.form['split'])
        size=size/100
        print(size)
        dataset=preprocessing(df)
        print(df)
        print(df.columns)
        dataset = dataset.loc[0:364]
        X=dataset.drop(['HGB','S. No.'],axis=1)
        y=dataset['HGB']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=52)
        print(y_test)

        return render_template('split.html',msg='Data Preprocessed and It Splits Succesfully')
    return render_template('split.html')

@app.route('/model',methods=['POST','GET'])
def model():
    if request.method=="POST":
        model=int(request.form['algo'])
        if model==0:
            return render_template('model.html',msg='Please Choose any Algorithm')
        elif model==1:
            model=Lasso()
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            score=r2_score(y_test,y_pred).round(4)
            score=score*100
            msg='The R2 Score for Lasso is ', score
            return render_template('model.html',msg=msg)

        else:
            model = Ridge()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred).round(4)
            score = score * 100
            msg = 'The R2 Score for Ridge is ', score
        return render_template('model.html',msg=msg)
    return render_template('model.html')

@app.route('/prediction',methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        val=request.form['d']
        # if f1=="male":
        #     val=0
        # else:
        #     val=1
        print(val)
        f2=request.form['age']
        f3=request.form['rbc']
        f4=request.form['pcv']
        f5=request.form['mcv']
        f6=request.form['mch']
        f7=request.form['mchc']
        f8=request.form['rdw']
        f9=request.form['tlc']
        f10=request.form['plt']
        print(f10)
        print(type(f10))
        l=[val,f2,f3,f4,f5,f6,f7,f8,f9,f10]
        model=Ridge()
        model.fit(X_train,y_train)
        ot=model.predict([l])
        print(ot)
        if ot<12:
            a='The Person is Diagnosed with Anemia'
        else:
            a="The Person is safe and not effected with Anemia"
        return render_template('prediction.html',msg=a)
    return render_template('prediction.html')

# @app.route('/homediabetes')
# def homediabetes():
#     return render_template('homediabetes.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/loaddiabetes',methods=['POST','GET'])
def loaddiabetes():
    global dataframe
    if request.method == "POST":
        file = request.files['file']
        print(type(file.filename))
        print('hi')
        dataframe = pd.read_csv(file)
        print(dataframe.head(2))
        return render_template('loaddiabetes.html', msg='Dataset Uploaded Successfully')
    return render_template('loaddiabetes.html')

@app.route('/viewdiabetes')
def viewdiabetes():
    print(dataframe)
    print(dataframe.head(2))
    print(dataframe.columns)
    return render_template('viewdiabetes.html',columns=dataframe.columns.values,rows=dataframe.values.tolist())

@app.route('/splitdiabetes',methods=["POST","GET"])
def splitdiabetes():
    global a,b,a_train,a_test,b_train,b_test
    for x in ['Insulin']:
        q75, q25 = np.percentile(dataframe.loc[:, x], [75, 25])
        intr_qr = q75 - q25

        max = q75 + (1.5 * intr_qr)
        min = q25 - (1.5 * intr_qr)

        dataframe.loc[dataframe[x] < min, x] = np.nan
        dataframe.loc[dataframe[x] > max, x] = np.nan
    for x in ['BMI']:
        q75, q25 = np.percentile(dataframe.loc[:, x], [75, 25])
        intr_qr = q75 - q25

        max = q75 + (1.5 * intr_qr)
        min = q25 - (1.5 * intr_qr)

        dataframe.loc[dataframe[x] < min, x] = np.nan
        dataframe.loc[dataframe[x] > max, x] = np.nan
    dataframe['Insulin'].fillna(dataframe['Insulin'].median(), inplace=True)
    dataframe['BMI'].fillna(dataframe['BMI'].median(), inplace=True)
    if request.method=="POST":
        size=int(request.form['split'])
        size=size/100
        print(size)
        print(dataframe)
        print(dataframe.columns)
        a = dataframe[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                  'BMI', 'DiabetesPedigreeFunction', 'Age']]
        b = dataframe[['Outcome']]
        a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=size,random_state=52)
        print(b_test)

        return render_template('splitdiabetes.html',msg='Data Preprocessed and It Splits Succesfully')
    return render_template('splitdiabetes.html')

@app.route('/modeldiabetes',methods=['POST','GET'])
def modeldiabetes():
    if request.method=="POST":
        model=int(request.form['algo'])
        if model==0:
            return render_template('modeldiabetes.html',msg='Please Choose any Algorithm')
        elif model==1:
            model=LogisticRegression()
            model.fit(a_train,b_train)
            b_pred=model.predict(a_test)
            score=accuracy_score(b_test,b_pred).round(4)
            score=score*100
            msg='The Accuracy Score for Logistic Regression is ', score
            return render_template('modeldiabetes.html',msg=msg)

        elif model==2:
            model = SVC()
            model.fit(a_train, b_train)
            b_pred = model.predict(a_test)
            score = accuracy_score(b_test, b_pred).round(4)
            score = score * 100
            msg = 'The Accuracy Score for SVM is ', score
            return render_template('modeldiabetes.html',msg=msg)
        elif model==3:
            model3 = Sequential()
            model3.add(Dense(16, activation='sigmoid'))
            model3.add(Dense(16, activation='sigmoid'))
            model3.add(Dense(64, activation='sigmoid'))
            model3.add(Dense(1, activation='sigmoid'))
            model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model3.fit(x=a_train, y=b_train, epochs=50)
            print('nn')
            pred3 = model3.predict(a_test)
            score3 = accuracy_score(b_test, pred3.round()).round(4)
            score3=score3*100
            msg = 'The Accuracy Score for Artificial Neural Network is ', score3
            return render_template('modeldiabetes.html', msg=msg)
        else:
            model4 = Sequential()
            model4.add(Dense(20, activation='sigmoid'))
            model4.add(Dense(16, activation='sigmoid'))
            model4.add(Dense(16, activation='sigmoid'))
            model4.add(Dense(94, activation='sigmoid'))
            model4.add(Dense(20, activation='sigmoid'))
            model4.add(Dense(16, activation='sigmoid'))
            model4.add(Dense(10, activation='sigmoid'))
            model4.add(Dense(10, activation='sigmoid'))
            model4.add(Dense(10, activation='sigmoid'))
            model4.add(Dense(10, activation='sigmoid'))
            model4.add(Dense(10, activation='sigmoid'))
            model4.add(Dense(10, activation='sigmoid'))
            model4.add(Dense(1, activation='sigmoid'))
            model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model4.fit(x=a_train, y=b_train, epochs=50)
            print('nn')
            pred4 = model4.predict(a_test)
            score4 = accuracy_score(b_test, pred4.round())
            score4 = score4 * 100
            score4.round(3)
            msg = 'The Accuracy Score for Deep Neural Network is ', score4
            return render_template('modeldiabetes.html', msg=msg)
    return render_template('modeldiabetes.html')

@app.route('/predictiondiab',methods=["POST","GET"])
def predictiondiab():
    print('aaaaaaaaaaaaaaaaa')
    if request.method=="POST":
        val=request.form['d']
        # if f1=="male":
        #     val=0
        # else:
        #     val=1
        print(val)
        print(type(val))
        f2=request.form['Pregnancies']

        f3=request.form['Glucose']
        f4=request.form['BloodPressure']
        f5=request.form['SkinThickness']
        f6=request.form['Insulin']
        f7=request.form['BMI']
        f8=request.form['DiabetesPedigreeFunction']
        f9=request.form['Age']
        print(f9)
        if val=="0":
            f2=0
            print(f2)
            l = [f2, f3, f4, f5, f6, f7, f8, f9]
            model = LogisticRegression()
            model.fit(a_train, b_train)
            ot = model.predict([l])
            print(ot)
            print('if')
        else:
            l = [f2, f3, f4, f5, f6, f7, f8, f9]
            model = Ridge()
            model.fit(a_train, b_train)
            ot = model.predict([l])
            print('else')
            print(ot)



        # f10=request.form['plt']
        # print(f10)
        # print(type(f10))
        # l=[val,f2,f3,f4,f5,f6,f7,f8,f9]
        # model=Ridge()
        # model.fit(X_train,y_train)
        # ot=model.predict([l])
        print(ot)
        if ot==1:
            a='The Person is Diagnosed with Diabetes'
        else:
            a="The Person is safe and not effected with Diabetes"
        return render_template('predictiondiab.html',msg=a)
    return render_template('predictiondiab.html')

if __name__=="__main__":
    app.run(debug=True)
