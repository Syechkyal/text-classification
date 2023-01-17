''''
import pathlib
import pandas as pd
from sklearn import svm
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from InformationGain import IGain
from ChiSquare import ChiSquare
from MutualInformation import MutualInformation
from TanpaSeleksiFitur import TanpaSeleksiFitur
from Prapengolahan import Preprocess 
import os
import re
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
import string
import time
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class Model:
    
    dataset = None
    model = None
    listData = []
    params_grid = {
     "C": [0.1],
    "kernel": ["linear"],
}

    
    
    def read_data(self,path):
        
        file = pathlib.Path(path)
        if file.exists():
            self.dataset = pd.read_excel(path)
        else:
            print('Path error : tidak ada data pada path ->',path)
    
    def prepare(self):
        if self.dataset is not None:
            print('Running Preprocessing...')
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            for text in self.dataset['pertanyaan']:
                pertanyaan = text.lower()
                noiseremoval = re.sub("[^a-zA-Z]", " ",pertanyaan) 
                katadasar = stemmer.stem(noiseremoval)
                tokenize = katadasar.split()
                self.listData.append(tokenize)
                       
            self.dataset = self.listData
            #print("panjang list data 1")
            #print(len(self.listData))
                   
        
        else :
            print('Input error : belum ada dataset')
            
        print('preprocess jalan.......')
        #for dataset in data_pc['pertanyaan']:
         #   print(self.dataset)
        self.dataset=pd.read_excel("Asset\preprocessed_data.xls")
      
    def tanpa_feature_selection(self):
        if self.dataset is not None:
            dataframe = self.dataset.copy()
            tanpa_selfit = TanpaSeleksiFitur(dataframe)
            return tanpa_selfit.process()
            
        else:
            print('Proses error : dataset tidak terdeteksi') 
            
    def feature_selectionIG(self,treshold):
        if self.dataset is not None:
            dataframe = self.dataset.copy()
            iGain = IGain(dataframe)
            iGain.process()
            return iGain.filtering(treshold)
        else:
            print('Proses error : dataset tidak terdeteksi')
           
    def feature_selectionCS(self,treshold):
        if self.dataset is not None:
            dataframe = self.dataset.copy()
            chiSquare = ChiSquare(dataframe)
            chiSquare.process()
            return chiSquare.filtering(treshold)
        else:
            print('Proses error : dataset tidak terdeteksi')
            
    def feature_selectionMI(self,treshold):
        if self.dataset is not None:
            dataframe = self.dataset.copy()
            mutualInformation= MutualInformation(dataframe)
            mutualInformation.process()
            return mutualInformation.filtering(treshold)
        else:
            print('Proses error : dataset tidak terdeteksi')
            
    def get_dataset(self):
        return self.dataset
    
    def classify(self,fitur):
        #Mencari Nilai TF
        TF = []
        
        print(len(fitur))
        for i in range (len(self.listData)):
          row =[]
          for j in range (len (fitur)):
            value = 0
            for k in range (len(self.listData[i])):
              if (fitur[j]== self.listData[i][k]):
                value = 1
            row.append(value)
          TF.append(row)
          
          
        DF = []
        IDF = []
        for j in range (len(fitur)):
          sum=0
          for i in range (len(self.listData)):
            sum = sum + TF[i][j]
          DF.append(sum)
          for k in range (len(self.listData)):
            x = len(self.listData)/sum
            value= math.log10(x)
          IDF.append(value)
          
        TFIDF = []

        for j in range (len(fitur)):
          row=[]
          for i in range (len(self.listData)):
            value = TF[i][j]*IDF[j]
            row.append(value)
          TFIDF.append(row)
        
        TFIDF=np.array(TFIDF)
        TFIDF=TFIDF.transpose()
        tfidf = pd.DataFrame(TFIDF,columns=fitur)
        print('tfidf selesai............')
        
        print('lanjut split data.......')
        
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(self.dataset['label'])
        
        
        X_train, X_test, y_train, y_test = train_test_split(tfidf, y, 
                                                                                    test_size=0.2, random_state=42)
        encoder.fit(y_train)
        Y_train = encoder.transform(y_train)
        
        encoder.fit(y_test)
        Y_test = encoder.transform(y_test)
        
        num_cols = X_train._get_numeric_data().columns
        print("Number of numeric features:",num_cols.size)
        
        names_of_predictors = list(X_train.columns.values)
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(Y_test)
        
        #tuning menggunakan search grid CV untuk mendadpatkan parameter terbaik
        start_time = time.time()
        svm_model = GridSearchCV(SVC(), self.params_grid, cv=10)
        svm_model.fit(X_train, y_train)
        
        #skor terbaik
        print('Best score for training data:', svm_model.best_params_,"\n") 
        
        #menyimpan hasil model terbaik
        final_model = svm_model.best_estimator_
        Y_pred = final_model.predict(X_test_scaled)
        Y_pred_label = list(encoder.inverse_transform(Y_pred))
        print(Y_pred)
        
        Acc = accuracy_score(Y_test, final_model.predict(X_test))
        Prec = precision_score(Y_test, final_model.predict(X_test), average = 'macro')
        Rec = recall_score(Y_test, final_model.predict(X_test), average = 'macro')
        F1 =  f1_score(Y_test, final_model.predict(X_test), average = 'macro')
        Cm = confusion_matrix(Y_test, final_model.predict(X_test))
        Tp = (Cm[0][0])+(Cm[1][1])+(Cm[2][2])
        Fn = (Cm[0][1])+(Cm[0][2])+(Cm[1][0])+(Cm[1][2])+(Cm[2][0])+(Cm[2][1])
        Fp = (Cm[1][0])+(Cm[2][0])+(Cm[0][1])+(Cm[2][1])+(Cm[0][2])+(Cm[1][2])
        Tn = (Cm[1][1])+(Cm[2][1])+(Cm[1][2])+(Cm[2][2])+(Cm[0][0])+(Cm[0][2])+(Cm[2][0])+(Cm[2][2])+(Cm[0][0])+(Cm[0][1])+(Cm[1][0])+(Cm[1][1])              
        Et = time.time() - start_time
        scores = []
        scores.append(['Uji ke','Tp','Fn','Fp','Tn','akurasi','precision','recall','f-measure','waktu Komputasi'])
        scores.append([len(fitur),Tp,Fn,Fp, Tn,Acc,Prec,Rec,F1,Et])
        self.model = final_model
        return scores
    
    
    '''                  
        print('Acc :',Acc)
        print('Pecision :',Prec)
        print('Recall :',Rec)
        #print(recall_score(Y_test, final_model.predict(X_test)))
        print('f1_score :',F1)
        print(Cm)
        print("\n")
        print(Et)
        print(classification_report(Y_test, final_model.predict(X_test)))
       ''' 
       ''''''
    def predicttanpaseleksifitur(self,text):
        datas = pd.Series(data = [text])
        datas = Preprocess.clean_symbol(datas)
        datas = Preprocess.clean_repeated_char(datas)
        datas = Preprocess.lowercase(datas)
        datas = Preprocess.stopword_removal(datas)
        datas = str(datas)
        string = []
        string.append(datas)
        print(string)
        
       
        prediction, raw_outputs = self.model.predict(string)
        print(prediction)
        if prediction == 'factoid':
            return "Pertanyaan Factoid"
        elif prediction == 'non-factoid':
             return "Pertanyaan Non-Factoid"
        else :
             return "Pertanayaan Others"
        
        
       
        

model = Model()
model.read_data("Asset\dataset_pertanyaan222.xlsx")
model.prepare()
fitur = model.tanpa_feature_selection()
score = model.classify(fitur)
pred = model.predicttanpaseleksifitur('siapa presiden Indonesia saat ini ?')

#print(model.prepare())


    