
import pandas as pd 
import pathlib
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize.treebank import TreebankWordDetokenizer




from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

class Preprocess:
    ## dokumen adalah tipe data series
    ## token adalah tipe data series yang sudah ditokenisasi
    
    def lowercase(dokumen):#method lowercase
        
        print('preprocess -> lowercase')#notifikasi apabila proses berjalan
        dokumen = dokumen.astype(str)  #mengubah dan memastikan isi dokumen berupa string
        dokumen = dokumen.str.lower()  #perintah merubah dokumen string tadi ke lowercase
        return dokumen                 #mengembalikan dokumen dalam bentuk lowercase
       
            
    def clean_symbol(dokumen): 
        print('preprocess -> clean symbol')
        return dokumen.apply(lambda text: re.sub(r"[^a-zA-Z0-9]+", ' ', text)) #
        print('input error : input is not series data or input is not correct, clean symbol stopped')
    
    
    def clean_repeated_char(dokumen):
        print('preprocess -> clean repeated char')
        return dokumen.apply(lambda text: re.sub(r'(.)\1{2,}', r'\1', text))
       
    
      
    
    def stemming(dokumen):
        ##prepare stemming
        factori = StemmerFactory()
        stemmer = factori.create_stemmer()
        
        print('proprocess -> stemming')
        return dokumen.apply(lambda text : stemmer.stem(text))
       
    
    
    @staticmethod
    def tokenize(dokumen):     
        return dokumen.apply(lambda text : nltk.word_tokenize(text))
       
    
    @staticmethod
    def untekonize(token):        
        return token.apply(lambda lis : str(TreebankWordDetokenizer().detokenize(lis)))
        
        

    