import math

import pandas as pd
from itertools import chain
class MutualInformation:
    
#membuat variabel global
    
    
    listData=[]
    unique_word=[]
    label=[]
    MI=[]
    token = None
    dataset = None
    fitur = []
    MI_fitur=[]
    
   
    
    def __init__(self,dataset):
        

        self.dataframe = dataset
        self.listData=[]
        for text in self.dataframe['pertanyaan']:
            self.token = text.split()
            self.listData.append(self.token)
        
        #print(self.listData)
        #print('panjang list data :')
        #print(len(self.listData))
       
        #print('panjang flatten list :')
        self.flatten_list = list(chain.from_iterable(self.listData))
        #print(len(self.flatten_list))
       
        #print(self.flatten_list)
    
        for i in self.flatten_list:
            if i not in self.unique_word:
                self.unique_word.append(i)     
       
                
        self.unique_word.sort()
        #print('panjang unique word :')
        #print(len(self.unique_word))
        #print(self.unique_word,"\n\n")
        
        
        for text in self.dataframe['label']:
            self.label.append(text)
        
        
        #print(self.label)
                
        print("Mutual Information jalan,")
        
    def process(self):
        #Nilai Kebenaran BB pada kelas Factoid
        BB01=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]==self.listData[j][k]):
                        if(self.label[j]=='factoid'):
                            count=count+1;
            BB01.append(count)

        #print(BB01[0])

        #Nilai Kebenaran BB pada kelas Non-Factoid
        BB02=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]==self.listData[j][k]):
                        if(self.label[j]=='non-factoid'):
                            count=count+1;
            BB02.append(count)
        
        #print(BB02[0])

        #Nilai Kebenaran BB pada kelas Other
        BB03=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]==self.listData[j][k]):
                        if(self.label[j]=='other'):
                            count=count+1;
            BB03.append(count)
        
        #print(BB03[0])
        
        #Nilai Kebenaran SB pada kelas Factoid
        SB01=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]!=self.listData[j][k]):
                        if(self.label[j]=='factoid'):
                            count=count+1;
            SB01.append(count)
        
        #print(SB01[0])
        
        #Nilai Kebenaran SB pada kelas Non-Factoid
        SB02=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]!=self.listData[j][k]):
                        if(self.label[j]=='non-factoid'):
                            count=count+1;
            SB02.append(count)
        
        #print(SB02[0])
        
        #Nilai Kebenaran SB pada kelas Other
        SB03=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]!=self.listData[j][k]):
                        if(self.label[j]=='other'):
                            count=count+1;
            SB03.append(count)
        
        #print(SB03[0])
        
        #Nilai Kebenaran BS pada kelas Factoid
        BS01=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]==self.listData[j][k]):
                        if(self.label[j]!='factoid'):
                            count=count+1;
            BS01.append(count)
        
        #print(BS01[0])
        
        #Nilai Kebenaran BS pada kelas Non-Factoid
        BS02=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]==self.listData[j][k]):
                        if(self.label[j]!='non-factoid'):
                            count=count+1;
            BS02.append(count)
        
        #print(BS02[0])
        
        #Nilai Kebenaran BS pada kelas Other
        BS03=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]==self.listData[j][k]):
                        if(self.label[j]!='other'):
                            count=count+1;
            BS03.append(count)
        
        #print(BS03[0])
        #Nilai Kebenaran SS pada kelas Factoid
        SS01=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]!=self.listData[j][k]):
                        if(self.label[j]!='factoid'):
                            count=count+1;
            SS01.append(count)
        
        #print(SS01[0])
        
        #Nilai Kebenaran SS pada kelas Non-Factoid
        SS02=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]!=self.listData[j][k]):
                        if(self.label[j]!='non-factoid'):
                            count=count+1;
            SS02.append(count)
        
        #print(SS02[0])
        
        #Nilai Kebenaran SS pada kelas Other
        SS03=[]
        for i in range (len(self.unique_word)):
            count=0;
            for j in range(len(self.listData)):
                for k in range (len(self.listData[j])):
                    if(self.unique_word[i]!=self.listData[j][k]):
                        if(self.label[j]!='other'):
                            count=count+1;
            SS03.append(count)
        
        #print(SS03[0])
        
        N = len(self.flatten_list)

        #N1. Kelas Factoid
        N1t01=[]
        for i in range (len(self.unique_word)):
            hasil = BS01[i]+BB01[i]
            N1t01.append(hasil)
        #print(N1t01[0])
        
        #N1. Kelas Non-Factoid
        N1t02=[]
        for i in range (len(self.unique_word)):
            hasil = BS02[i]+BB02[i]
            N1t02.append(hasil)
        #print(N1t02[0])
        
        #N1. Kelas Other
        N1t03=[]
        for i in range (len(self.unique_word)):
            hasil = BS03[i]+BB03[i]
            N1t03.append(hasil)
        #print(N1t03[0])
        
        #N1 Kelas Factoid
        N101=[]
        for i in range (len(self.unique_word)):
            hasil = SB01[i]+BB01[i]
            N101.append(hasil)
        #print(N101[0])
        
        #N1 Kelas Non-Factoid
        N102=[]
        for i in range (len(self.unique_word)):
            hasil = SB02[i]+BB02[i]
            N102.append(hasil)
        #print(N102[0])
        
        #N1 Kelas Other
        N103=[]
        for i in range (len(self.unique_word)):
            hasil = SB03[i]+BB03[i]
            N103.append(hasil)
        #print(N103[0])
        
        #N0t Kelas Factoid
        N0t01=[]
        for i in range (len(self.unique_word)):
            hasil = SB01[i]+SS01[i]
            N0t01.append(hasil)
        #print(N0t01[0])
        
        #N0t Kelas Non-Factoid
        N0t02=[]
        for i in range (len(self.unique_word)):
            hasil = SB02[i]+SS02[i]
            N0t02.append(hasil)
        #print(N0t02[0])
        
        #N0t Kelas Other
        N0t03=[]
        for i in range (len(self.unique_word)):
            hasil = SB03[i]+SS03[i]
            N0t03.append(hasil)
        #print(N0t03[0])
        
        #N0 Kelas Factoid
        N001=[]
        for i in range (len(self.unique_word)):
            hasil = BS01[i]+SS01[i]
            N001.append(hasil)
        #print(N001[0])
        
        #N0 Kelas Non-Factoid
        N002=[]
        for i in range (len(self.unique_word)):
            hasil = BS02[i]+SS02[i]
            N002.append(hasil)
        #print(N002[0])
        
        #N0 Kelas Other
        N003=[]
        for i in range (len(self.unique_word)):
            hasil = BS03[i]+SS03[i]
            N003.append(hasil)
        #print(N003[0])
        
        #Mutual Informatin Kelas Factoid
        MI01=[]
        
        for i in range(len(self.unique_word)):
            #proses 1 
            hasil1_1 = (N*BB01[i])/(N1t01[i]*N101[i])
            if(hasil1_1 == 0):
                hasil1_2 = 0
            else:
                hasil1_2=math.log2(hasil1_1)
            hasil1 = (BB01[i]/N)*hasil1_2
            
            #proses 2
            hasil2_1 = (N*SB01[i])/(N0t01[i]*N101[i])
            if(hasil2_1 == 0):
                hasil2_2 = 0
            else:
                hasil2_2=math.log2(hasil2_1)
            hasil2 = (SB01[i]/N)*hasil2_2
            
            #proses 3
            hasil3_1 = (N*BS01[i])/(N1t01[i]*N001[i])
            if(hasil3_1 == 0):
                hasil3_2 = 0
            else:
                hasil3_2=math.log2(hasil3_1)
            hasil3 = (BS01[i]/N)*hasil3_2
            
            #proses 4
            hasil4_1 = (N*SS01[i])/(N0t01[i]*N001[i])
            if(hasil4_1 == 0):
                hasil4_2 = 0
            else:
                hasil4_2=math.log2(hasil4_1)
            hasil4 = (SS01[i]/N)*hasil4_2
            
            hasil = hasil1+hasil2+hasil3+hasil4
            MI01.append(hasil)
        
        #print(MI01[0])
        #Mutual Informatin Kelas Non-Factoid
        MI02=[]
        
        for i in range(len(self.unique_word)):
            #proses 1 
            hasil1_1 = (N*BB02[i])/(N1t02[i]*N102[i])
            if(hasil1_1 == 0):
                hasil1_2 = 0
            else:
                hasil1_2=math.log2(hasil1_1)
            hasil1 = (BB02[i]/N)*hasil1_2
            
            #proses 2
            hasil2_1 = (N*SB02[i])/(N0t02[i]*N102[i])
            if(hasil2_1 == 0):
                hasil2_2 = 0
            else:
                hasil2_2=math.log2(hasil2_1)
            hasil2 = (SB02[i]/N)*hasil2_2
            
            #proses 3
            hasil3_1 = (N*BS02[i])/(N1t02[i]*N002[i])
            if(hasil3_1 == 0):
                hasil3_2 = 0
            else:
                hasil3_2=math.log2(hasil3_1)
            hasil3 = (BS02[i]/N)*hasil3_2
            
            #proses 4
            hasil4_1 = (N*SS02[i])/(N0t02[i]*N002[i])
            if(hasil4_1 == 0):
                hasil4_2 = 0
            else:
                hasil4_2=math.log2(hasil4_1)
            hasil4 = (SS02[i]/N)*hasil4_2
            
            hasil = hasil1+hasil2+hasil3+hasil4
            MI02.append(hasil)
        
        #print(MI02[0])
        
        #Mutual Informatin Kelas Other
        MI03=[]
        
        for i in range(len(self.unique_word)):
            #proses 1 
            hasil1_1 = (N*BB03[i])/(N1t03[i]*N103[i])
            if(hasil1_1 == 0):
                hasil1_2 = 0
            else:
                hasil1_2=math.log2(hasil1_1)
            hasil1 = (BB03[i]/N)*hasil1_2
            
            #proses 2
            hasil2_1 = (N*SB03[i])/(N0t03[i]*N103[i])
            if(hasil2_1 == 0):
                hasil2_2 = 0
            else:
                hasil2_2=math.log2(hasil2_1)
            hasil2 = (SB03[i]/N)*hasil2_2
            
            #proses 3
            hasil3_1 = (N*BS03[i])/(N1t03[i]*N003[i])
            if(hasil3_1 == 0):
                hasil3_2 = 0
            else:
                hasil3_2=math.log2(hasil3_1)
            hasil3 = (BS03[i]/N)*hasil3_2
            
            #proses 4
            hasil4_1 = (N*SS03[i])/(N0t03[i]*N003[i])
            if(hasil4_1 == 0):
                hasil4_2 = 0
            else:
                hasil4_2=math.log2(hasil4_1)
            hasil4 = (SS03[i]/N)*hasil4_2
            
            hasil = hasil1+hasil2+hasil3+hasil4
            MI03.append(hasil)
        
        #print(MI03[0])
        
        
        for i in range(len(self.unique_word)):
            nilai = max(MI01[i],MI02[i],MI03[i])
            self.MI.append(nilai)
        #print(MI,"\n\n")
        
    #method penyaring dan memperbaru dataset dengan treshold masukan oleh pengguna    
    def filtering(self,treshold):
        fitur = []
        MI_fitur=[]
        miData=[]
        
        print('-------Mutual Information SCORE------')
        print(self.MI)
        
        for i in range (len(self.unique_word)):
            if(self.MI[i]>=treshold):
                fitur.append(self.unique_word[i])
                MI_fitur.append(self.MI[i])
                
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
            miData.append(value)    
        
         
        df_mi = {
            "pertanyaan" : miData,
            "label" : self.dataframe['label']
        }

        df_mi = pd.DataFrame(df_mi)
        return(df_mi)
    
        '''
        total_term = len(self.unique_word)
        term_MI = len(self.fitur)
    
        print("Nilai Treshold : ",treshold)
        print("Total term : ",total_term)
        print("Total term setelah MI : ",term_MI)
        print("Selisih : ",total_term-term_MI)
        '''
        
        '''       
model=pd.read_excel("Asset\preprocessed_data.xls")
coba=MutualInformation(model)
coba.process()
coba.filtering(0.00023)
print(coba)
'''