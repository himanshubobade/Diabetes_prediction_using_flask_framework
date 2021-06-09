from flask import Flask, abort, jsonify, request, render_template
#from sklearn.externals.joblib import joblib
import joblib
import numpy as np
import json
import pickle
app = Flask(__name__)
# Load the model
modelrf = pickle.load(open('data/RandomForest.pkl','rb'))
modeldt = pickle.load(open('data/DecisionTree.pkl','rb'))
modelknn = pickle.load(open('data/KNN.pkl','rb'))
modellda = pickle.load(open('data/LDA.pkl','rb'))
modelsvm = pickle.load(open('data/SVM.pkl','rb'))
modellr = pickle.load(open('data/LogisticRegressionModel.pkl','rb'))

@app.route('/')
def home():
   return render_template('form.html')

@app.route('/analysis',methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        preg= request.form['preg']

        gluc= request.form['gluc']

        blood_pressure= request.form['blood_pressure']

        skin_th= request.form['skin_th']

        insln= request.form['insln']

        b_m_i= request.form['b_m_i']

        d_p_func = request.form['d_p_func']
      
        AGE = request.form['AGE']
        
        sample_data = [preg,gluc,blood_pressure,skin_th,insln, b_m_i,d_p_func,AGE]
        clean_data = [float(i) for i in sample_data]
        print(clean_data)
        ex = np.array(clean_data).reshape(1,-1)
        print(ex)
        result_prediction = modelrf.predict(ex)
        print(result_prediction)

        result_dt = modeldt.predict(ex)
        result_knn = modelknn.predict(ex)
        result_lda = modelknn.predict(ex)
        result_lr = modellr.predict(ex)
        result_svm = modelsvm.predict(ex)

        return render_template('analysis.html',result_prediction=result_prediction,preg=preg,blood_pressure=blood_pressure,gluc=gluc,skin_th=skin_th,insln=insln,b_m_i=b_m_i,d_p_func=d_p_func,AGE=AGE,result_dt=result_dt,result_knn=result_knn,result_lda=result_lda,result_lr=result_lr,result_svm=result_svm)
        


if __name__ == '__main__':
    app.run(port=4000, debug=True)
