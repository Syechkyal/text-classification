import csv
import pathlib
import pandas as pd
from sklearn import svm
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from InformationGain import IGain
from ChiSquare import ChiSquare
from MutualInformation import MutualInformation
#from TanpaSeleksiFitur import TanpaSeleksiFitur
from Datapreprocessing import Preprocess 
import os
import re
import nltk
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
import xlwt


class Proses:
    
    dataset = None
    model = None
    listData = []
   

    
    
    def input_data(self,path):
        
        file = pathlib.Path(path)
        if file.exists():
            self.dataset = pd.read_excel(path)
        else:
            print('Path error : tidak ada data pada path ->',path)
    
    def praprosses_data(self):
        if self.dataset is not None:
            print('Running Preprocessing...')
            data_pc = self.dataset
            data_pc['pertanyaan'] = data_pc['pertanyaan'].apply(lambda text: re.sub(r"[^a-zA-Z0-9]+", ' ', text))
            data_pc['pertanyaan'] = data_pc['pertanyaan'].apply(lambda text: re.sub(r'(.)\1{2,}', r'\1', text))          
            data_pc['pertanyaan'] = data_pc['pertanyaan'].apply(lambda text : nltk.word_tokenize(text))
            #data_pc['pertanyaan'] = Preprocess.stemming(data_pc['pertanyaan'])
         
            
            
        return data_pc
           
    
            
        
        #self.dataset=pd.read_excel("Asset\preprocessed_data.xls")
         
    def tanpa_feature_selection(self):
        if self.dataset is not None:
            dataframe = self.dataset.copy()
            return dataframe
            
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
    
    def get_total_feature(self,dataset):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset['pertanyaan'].values.astype('U'))
        return X.shape[1]
    
    def classify(self,dataset,kernel_clf,c_clf):
        print(dataset)
        print(kernel_clf)
        print(c_clf)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        vectorizer = TfidfVectorizer(use_idf=True)
        X = vectorizer.fit_transform(dataset['pertanyaan'].values.astype('U'))
        Y = dataset['label']
        SVM = svm.SVC(C=10, kernel='rbf')
        scores = []
        scores.append(['Uji ke','tp','fp','tn','fn','akurasi','precision','recall','f-measure','waktu Komputasi'])
        cv = KFold(n_splits=10)
        index_hasil = 1
        for train_index, test_index in cv.split(X):
            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            start_time = time.time()
            SVM.fit(X_train,Y_train) #training
            Y_pred = SVM.predict(X_test) #test
            Cm = confusion_matrix(Y_test, Y_pred)
            Tp =(Cm[0][0])+(Cm[1][1])+(Cm[2][2])
            Fn = (Cm[0][1])+(Cm[0][2])+(Cm[1][0])+(Cm[1][2])+(Cm[2][0])+(Cm[2][1])
            Fp = (Cm[1][0])+(Cm[2][0])+(Cm[0][1])+(Cm[2][1])+(Cm[0][2])+(Cm[1][2])   
            Tn = (Cm[1][1])+(Cm[2][1])+(Cm[1][2])+(Cm[2][2])+(Cm[0][0])+(Cm[0][2])+(Cm[2][0])+(Cm[2][2])+(Cm[0][0])+(Cm[0][1])+(Cm[1][0])+(Cm[1][1])            
            acc = round(accuracy_score(Y_test,Y_pred),2)
            prec = round(precision_score(Y_test,Y_pred, average = 'macro'),2)
            rec = round(recall_score(Y_test,Y_pred, average = 'macro'),2)
            f1 = round(f1_score(Y_test,Y_pred, average = 'macro'),2)
            #f1 = f1_score(Y_test,Y_pred)
            execution_time = round((time.time() - start_time),2)
            scores.append([index_hasil,Tp,Fn,Fp,Tn,acc,prec,rec,f1,execution_time])
            index_hasil +=1
            
        temp = ['Rata-rata',0,0,0,0,0,0,0,0,0]
        for i in range(1,11):
            for j in range(1,10):
                temp[j] += scores[i][j]
        
        for i in range(1,10):
            temp[i] = round((temp[i]/10),2)
                
        scores.append(temp)
        print(scores)
        '''
        with open("assets/hasil_evaluasi.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(scores) 
            '''
        self.model = SVM
        return scores
    
    

    def predicttanpaseleksifitur(self,text):
        datas = pd.Series(data = [text])
        datas = Preprocess.clean_symbol(datas)
        datas = Preprocess.clean_repeated_char(datas)
        datas = Preprocess.lowercase(datas)
        datas = Preprocess.stopword_removal(datas)
        
        vectorizer = TfidfVectorizer(use_idf=True)
        
        TF_IDF = vectorizer.fit(self.dataset['pertanyaan'].values.astype('U'))
        datas = TF_IDF.transform(datas.values.astype('U'))
        
        prediction = self.model.predict(datas)
        print(prediction)
        if prediction == 'factoid':
            return "Pertanyaan Factoid"
        elif prediction == 'non-factoid':
             return "Pertanyaan Non-Factoid"
        else :
             return "Pertanyaan Others"
    
    def predictseleksifiturIG(self,text):
        datas = pd.Series(data = [text])
        datas = Preprocess.clean_symbol(datas)
        datas = Preprocess.clean_repeated_char(datas)    
        datas = Preprocess.lowercase(datas)
        datas= Preprocess.stemming(datas)
        datas = Preprocess.stopword_removal(datas)
        
        
        vectorizer = TfidfVectorizer(use_idf=True)
        read=pd.read_excel('Asset/datasethasilIG.xlsx')
        TF_IDF = vectorizer.fit(read['pertanyaan'].values.astype('U'))
        datas = TF_IDF.transform(datas.values.astype('U'))
        
        prediction = self.model.predict(datas)
        print(prediction)
        
        if prediction == 'factoid':
            return "Pertanyaan Factoid"
        elif prediction == 'non-factoid':
             return "Pertanyaan Non-Factoid"
        else :
             return "Pertanyaan Others"
        
    def predictseleksifiturCS(self,text):
            datas = pd.Series(data = [text])
            datas = Preprocess.clean_symbol(datas)
            datas = Preprocess.clean_repeated_char(datas)    
            datas = Preprocess.lowercase(datas)
            datas= Preprocess.stemming(datas)
            datas = Preprocess.stopword_removal(datas)
            
            
            vectorizer = TfidfVectorizer(use_idf=True)
            read=pd.read_excel('Asset/datasethasilCS.xlsx')
            TF_IDF = vectorizer.fit(read['pertanyaan'].values.astype('U'))
            datas = TF_IDF.transform(datas.values.astype('U'))
            
            prediction = self.model.predict(datas)
            print(prediction)
            
            if prediction == 'factoid':
                return "Pertanyaan Factoid"
            elif prediction == 'non-factoid':
                 return "Pertanyaan Non-Factoid"
            else :
                 return "Pertanyaan Others"
             
    def predictseleksifiturMI(self,text):
                datas = pd.Series(data = [text])
                datas = Preprocess.clean_symbol(datas)
                datas = Preprocess.clean_repeated_char(datas)    
                datas = Preprocess.lowercase(datas)
                datas= Preprocess.stemming(datas)
                datas = Preprocess.stopword_removal(datas)
                
                
                vectorizer = TfidfVectorizer(use_idf=True)
                read=pd.read_excel('Asset/datasethasilMI.xlsx')
                TF_IDF = vectorizer.fit(read['pertanyaan'].values.astype('U'))
                datas = TF_IDF.transform(datas.values.astype('U'))
                
                prediction = self.model.predict(datas)
                print(prediction)
                
                if prediction == 'factoid':
                    return "Pertanyaan Factoid"
                elif prediction == 'non-factoid':
                     return "Pertanyaan Non-Factoid"
                else :
                     return "Pertanyaan Others"
        
        
       
'''

model = Model()
model.read_data("Asset\dataset_pertanyaan222.xlsx")
model.prepare()
#fitur = model.feature_selectionMI(0.00023)
fitur = model.feature_selectionIG(1.667)
#fitur = model.feature_selectionCS(2.5)
#fitur = model.tanpa_feature_selection()
score = model.classify(fitur,'rbf',10)
pred = model.predictseleksifitur('apa kau mau semua harta bendamu ludes begitu saja karena kau selalu saja memboros ?',fitur)

#print(model.prepare())

'''
    