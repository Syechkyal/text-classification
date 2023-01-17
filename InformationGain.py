import math
import pandas as pd
import csv
from itertools import chain
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
class InformationGain:
    #membuat variabel global
    unique_word = []
    f_freq = []
    n_freq = []
    o_freq = []
    token = None
    dataset = None
    entropy = []
    f_ent = []
    n_ent = []
    o_ent = []
    IG=[]
    total_term = 0
    term_ig =[]

    def __init__(self,dataframe):
        self.dataset = dataframe
        self.token = (dataframe['pertanyaan']).apply(lambda text : nltk.word_tokenize(text))
        
        ## Inisialisasi unique_word
        for sentence in self.token:
            for term in sentence :
                if term in self.unique_word :
                    continue;
                else :
                    self.unique_word.append(term)
        
        
        
        
                    
        self.f_freq = [0]*len(self.unique_word)
        self.n_freq = [0]*len(self.unique_word)
        self.o_freq = [0]*len(self.unique_word)
        self.total_term=(len(self.unique_word))
        
        #print(len(self.unique_word))
        print("Igainjalan,")
        
    def process(self):
        print('--- Information Gain Process ---')
        ## Hitung total frekuensi unique_word  factoid, non factoid dan other
        df_f = self.dataset[self.dataset['label']=='factoid']
        df_n = self.dataset[self.dataset['label']=='non-factoid']
        df_o = self.dataset[self.dataset['label']=='other']
        print(len(df_f))
        
        for i,term in enumerate(self.unique_word):
        ## mencari jumlah kemunculan term didokumen factoid
            for pertanyaan in df_f['pertanyaan']:
                if term in pertanyaan:
                    self.f_freq[i] +=1
            ## mencari jumlah kemunculan term didokumen non-factoid
            for pertanyaan in df_n['pertanyaan']:
                if term in pertanyaan:
                    self.n_freq[i] +=1 
            ## mencari jumlah kemunculan term didokumen other
            for pertanyaan in df_o['pertanyaan']:
                if term in pertanyaan:
                    self.o_freq[i] +=1 
        
        def imlog(a,b):
            if a == 0 or b==0:
                return 0
            else:
                return math.log2(a/b)
        
        ##hitung entropy
        entro_set = 1.644
        
    
        for i,term in enumerate(self.unique_word):
            #mencari entropy factoid.
            #=((-jumlah kemunculan term di dokumen/jumlah dokumen)IMLOG2(Sama)+(-Jumlah ketidakmunculan term pada dokumen/jumlah dokumen)IMLOG2(sama)
             f_temp = -(((self.f_freq[i]/len(df_f))*imlog(self.f_freq[i],len(df_f)))+(((519-self.f_freq[i])/len(df_f))*imlog((519-self.f_freq[i]),len(df_f))))
             #mencari entropy non-factoid
             n_temp = -(((self.n_freq[i]/len(df_n))*imlog(self.n_freq[i],len(df_n)))+(((491-self.n_freq[i])/len(df_n))*imlog((491-self.n_freq[i]),len(df_n))))
             #mencari entropy other
             o_temp = -(((self.o_freq[i]/len(df_o))*imlog(self.o_freq[i],len(df_o)))+(((185-self.o_freq[i])/len(df_o))*imlog((185-self.o_freq[i]),len(df_o))))
             
             #mencari total entropy 
             #(jumlah sample data partisi J atau jumlah dokumen positif/jumlah dokumen)*entropy factoid+ (jumlah sample data partisi J atau jumlah dokumen non factoid/jumlah dokumen)
             total_entro = -((519/1195)*f_temp) - ((491/1195)*n_temp) - ((185/1195)*o_temp)
             #menghitung Ig yakni entropi set - total entropi
             ig = (entro_set-total_entro)
             #selanjutnya di append ke variabel global untuk melihat hasil nya
             self.entropy.append(total_entro)

             self.f_ent.append(f_temp)
             self.n_ent.append(n_temp)
             self.o_ent.append(o_temp)
             self.IG.append(ig)
      
        
             
    
    #method penyaring dan memperbaru dataset dengan treshold masukan oleh pengguna
    def filtering(self,treshold):
        print('-----SCORE INFORMATION GAIN-----')
        print(self.IG)
        for i,term in enumerate(self.unique_word): #melakukan looping dan mengecek jika IG term melebihi dan sama dengan treshold
            if self.IG[i] >= treshold:            #maka term tidak dihapus jika tak memenuhi syarat maka term akan di remove.
                continue;
                print(term)
            else:
                for j,item in enumerate(self.token) :
                    if term in item:
                        self.token[j].remove(term)
        output_IG = self.dataset
        output_IG['pertanyaan'] = self.token.apply(lambda lis : str(TreebankWordDetokenizer().detokenize(lis)))
        print(output_IG)
        
        #output_IG.to_excel("Asset\hasilseleksiIG.xls")
        
        return(output_IG)
        '''
        #output = self.dataset 
        #output['pertanyaan'] = Preprocess.untekonize(output_IG)
        
        #print(len(self.token))
        #print(output)
        #print(self.IG)
        #print(len(self.IG))
        flatten_list = list(chain.from_iterable(output_IG))
        #print(len(flatten_list))
        
        
        unique_word=[]
        
        for i in flatten_list:
            if i not in unique_word:
                unique_word.append(i)
        
        unique_word.sort()        
        self.term_ig= unique_word
        return self.term_ig
        
               
        
        #return(self.term_ig)
        
        #len_ig=(len(self.term_ig))
        #print(len(self.term_ig))
        #print(term_ig,"\n\n")
        
        '''
        '''
        print("Nilai Treshold : ",treshold)
        print("Total term : ",  self.total_term)
        print("Total term setelah ig : ",len_ig)
        print("Selisih : ",self.total_term-len_ig)
        return(output_IG)
    '''
'''
model=pd.read_excel("Asset\preprocessed_data.xls")
coba=IGain(model)
coba.process()
coba.filtering(1.667)
print(coba)
'''
       
