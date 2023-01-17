import math

import pandas as pd
from itertools import chain
class ChiSquare:
    
    #membuat variabel global
    
    
    listData=[]
    lData=[]
    unique_word=[]
    label=[]
    chi_square=[]
    token = None
    dataset = None
    fitur = []
    CS_fitur=[]
    dataframe = None
    
   
    
    def __init__(self,dataset):
        self.dataframe = dataset
        self.listData=[]
        print(self.listData)
        for text in self.dataframe['pertanyaan']:
            self.token = text.split()
            self.listData.append(self.token)
        print(len(self.listData))
    
        
        self.flatten_list = list(chain.from_iterable(self.listData))
        
        
        for i in self.flatten_list:
            if i not in self.unique_word:
                self.unique_word.append(i)     
       
                
        self.unique_word.sort()
        
        
        
        for text in self.dataframe['label']:
            self.label.append(text)
       
    def process(self):
        A0=[]
        B0=[]
        C0=[]
        D0=[]
        
        print('--- Chi Square Process ---')
        #Hitung Nilai ABCD Factoid
        for i in range(len(self.unique_word)):
            countA=0
            countB=0    
            countC=0
            countD=0
            
            for j in range(len(self.listData)):
                
                for k in range(len(self.listData[j])):
                    
                    if(self.unique_word[i] == self.listData[j][k]):
                       
                        if(self.label[j]=='factoid'):
                            countA= countA+1
                            
                        else:
                            countB= countB+1
            
                            
            A0.append(countA)
            B0.append(countB)
    
            for j in range(len(self.listData)):
                cekC=True
                cekD=True
                if(self.label[j]=='factoid'):
                    for k in range (len(self.listData[j])):
                        if(self.unique_word[i]==self.listData[j][k]):
                            cekC=False
                    if(cekC==True):
                        countC=countC+1
                else:
                    for k in range (len(self.listData[j])):
                        if(self.unique_word[i]==self.listData[j][k]):
                            cekD=False
                    if(cekD==True):
                        countD=countD+1   
                        
            C0.append(countC)
            D0.append(countD)
        
        
        
        #Hitung Nilai ABCD Non-Factoid
        A1=[]
        B1=[]
        C1=[]
        D1=[]
        for i in range(len(self.unique_word)):
            countA=0
            countB=0    
            countC=0
            countD=0 
            for j in range(len(self.listData)):
                for k in range(len(self.listData[j])):
                    if(self.unique_word[i] == self.listData[j][k]):
                        if(self.label[j]=='non-factoid'):
                            countA=countA+1
                        else:
                            countB= countB+1
            A1.append(countA)
            B1.append(countB)
        
            for j in range(len(self.listData)):
                    cekC=True
                    cekD=True
                    if(self.label[j]=='non-factoid'):
                        for k in range (len(self.listData[j])):
                            if(self.unique_word[i]==self.listData[j][k]):
                                cekC=False
                        if(cekC==True):
                            countC=countC+1
                    else:
                        for k in range (len(self.listData[j])):
                            if(self.unique_word[i]==self.listData[j][k]):
                                cekD=False
                        if(cekD==True):
                            countD=countD+1   
            C1.append(countC)
            D1.append(countD)
            
       
        
        
        #Hitung Nilai ABCD Others
        A2=[]
        B2=[]
        C2=[]
        D2=[]
        for i in range(len(self.unique_word)):
            countA=0
            countB=0    
            countC=0
            countD=0 
            for j in range(len(self.listData)):
                for k in range(len(self.listData[j])):
                    if(self.unique_word[i] == self.listData[j][k]):
                        if(self.label[j]=='other'):
                            countA=countA+1
                        else:
                            countB= countB+1
            A2.append(countA)
            B2.append(countB)
            
            for j in range(len(self.listData)):
                cekC=True
                cekD=True
                if(self.label[j]=='other'):
                    for k in range (len(self.listData[j])):
                        if(self.unique_word[i]==self.listData[j][k]):
                            cekC=False
                    if(cekC==True):
                        countC=countC+1
                else:
                    for k in range (len(self.listData[j])):
                        if(self.unique_word[i]==self.listData[j][k]):
                            cekD=False
                    if(cekD==True):
                        countD=countD+1   
            C2.append(countC)
            D2.append(countD)
            
        
        
        
        #CHI SQUARE Kelas FACTOID
        sumDoc=len(self.listData)
        NN0=[]
        CS0=[]
        for i in range(len(self.unique_word)):
            hasil = (A0[i]+C0[i])*(B0[i]+D0[i])*(A0[i]+B0[i])*(C0[i]+D0[i])
            NN0.append(hasil)
    
        for i in range(len(self.unique_word)):
            hasil = (sumDoc*((A0[i]*D0[i])-(C0[i])*B0[i])**2)/NN0[i]
            CS0.append(hasil)
        
        #CHI SQUARE Kelas NON-FACTOID
        NN1=[]
        CS1=[]
        for i in range(len(self.unique_word)):
            hasil = (A1[i]+C1[i])*(B1[i]+D1[i])*(A1[i]+B1[i])*(C1[i]+D1[i])
            NN1.append(hasil)
    
        for i in range(len(self.unique_word)):
            hasil = (sumDoc*((A1[i]*D1[i])-(C1[i])*B1[i])**2)/NN1[i]
            CS1.append(hasil)
        
        #CHI SQUARE Kelas OTHER
        NN2=[]
        CS2=[]
        for i in range(len(self.unique_word)):
            hasil = (A2[i]+C2[i])*(B2[i]+D2[i])*(A2[i]+B2[i])*(C2[i]+D2[i])
            NN2.append(hasil)
            
        for i in range(len(self.unique_word)):
            hasil = (sumDoc*((A2[i]*D2[i])-(C2[i])*B2[i])**2)/NN2[i]
            CS2.append(hasil)
        
        #Menentukan Hasil Chi-Square Terbesar
        
        for i in range(len(self.unique_word)):
            nilai = max(CS0[i],CS1[i],CS2[i])
            self.chi_square.append(nilai)
        
      
        print('process selesai..')
            
   
         
    #method penyaring dan memperbaru dataset dengan treshold masukan oleh pengguna
    def filtering(self,treshold):      
        fitur = []
        CS_fitur=[]
        csData=[]
        print('----------SCORE CHI SQUARE------------')
        print(self.chi_square)

     
        for i in range (len(self.unique_word)):
            if(self.chi_square[i]>=treshold):
                fitur.append(self.unique_word[i])
                CS_fitur.append(self.chi_square[i])
                
        for i in range (len(self.listData)):
            value=''
            for j in range (len(self.listData[i])):
                for x in range (len(fitur)):
                    if(self.listData[i][j]==fitur[x]):
                        if(value == ''):
                            value = self.listData[i][j]
                        else:
                            value = value + '//' + self.listData[i][j]
            value = value.replace('//',' ')
            csData.append(value)    
            
     
        
        df_cs = {
            "pertanyaan" : csData,
            "label" : self.dataframe['label']
        }

        df_cs = pd.DataFrame(df_cs)
        return(df_cs)
        
    
            
        
        
       
            
        
    
    
    
    
