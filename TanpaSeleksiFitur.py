'''
import math
from Prapengolahan import Preprocess
import pandas as pd
from itertools import chain
class TanpaSeleksiFitur:
    
    listData=[]
    unique_word=[]
    label=[]  
    token = None
    dataset = None
    fitur = []
    
    
   
    
    def __init__(self,dataframe):
        
      
        for text in dataframe['pertanyaan']:
            token = text.split()
            self.listData.append(token)
        
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
        
    def process(self):
        print('--- Tanpa Seleksi Fitur Process ---')
        output_TS = self.unique_word
        
        return output_TS
    '''