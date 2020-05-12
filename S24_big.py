#   New Wordlist form University Hanover
#   cool stuff from Christian Wartena
#   TF-IDF deaktiviert
#   Polarity Index PI : ... [ -1 ; +1 ]
#   Neue Indices SI & EI definiert
#   TK1: TextKorpus komplett (ohne Sonderzeichen & Emojis etc.)
#   TK2: TextKorpus ohne dtsch. Stoppwörter  (der,die,das,...)
#   TK3: TextKorpus Anzahl Einzelwörter (unique words) mit Frequenz (Häufigkeitszahl)
#   TK4: TextKorpus  Sentiment Wörter (pos + neg Polarität)
#    EK: Summe aller Emotionen in TK4
#    SI: Sentiment Index = TK4 / TK1   ... [ 0 ; 2 ] 
#    EI: Emotion Index   =  EK / TK2   ... [ 0 ; 8 ]
#   
#   copyright 2020, Peter Dittrich
#
from HanTa import HanoverTagger as ht
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
from textblob_de import TextBlobDE as tbde
import os,math,re,sys
from textblob_de import PatternParser
from textblob_de.lemmatizers import PatternParserLemmatizer
from termcolor import colored
_lemmatizer = PatternParserLemmatizer()
#
#pd.set_option("display.max_columns",30)
pd.set_option('display.max_columns', None)

print("........................................................................")
#
lenni = len(sys.argv)
if lenni == 1: 
   print("  ACHTUNG Eingabefehler   ")
   sys.exit(" No Corpus file in command line")
text = sys.argv[1]
print("      >>>>>>>  Auswertung CORPUS : ",text,"  <<<<<<<<< ")
print(".")
tsname = text
now = datetime.datetime.now()
date = now.strftime('%d-%m-%Y')
zeit = now.strftime('%H:%M')
#
#tsname="xxxxx"
# tsname ="TAZ_reportage.txt"
#print(" Current dir :",os.getcwd())
print("      Datum: ",date,"  Zeit: ", zeit," Uhr")

print("........................................................................")
class Taz_Exploration:
    def __init__(self, data, my_stopwords):
        """
        Die wichtigsten Files sind hier
        """
        self.data_file = data
        self.my_stopword = my_stopwords
        self.stopwords_list = []
        self.nlines = []
        self.clean_data= []
        self.blob = None
        self.sliste_n = ()
        self.encoded_vals = []
        self.labels = {}
        self.wörter_org = []
        self.my_dict = {}
        self.encoded_vals = []
        self.hanta_lemma = [] 

    def load_stpwords(self, stopwords_de):
        """
        Ladet die Stopwords list
        """
        g = open(stopwords_de,'r',encoding="utf8")
        read_g = g.readlines()
        for x in read_g:
            sub_string = re.sub(r"(?<=[a-z])\r?\n","", x)
            stp_word = sub_string.replace("\n","")
            self.stopwords_list.append(stp_word)
        self.stopwords_list = self.stopwords_list + self.my_stopword 
        return self.stopwords_list

    def open_clean_file(self,file):
        """
        Öffnet file und macht ihn fresh & clean
        """
        f = open(tsname,'r',encoding="utf8")
        lines = f.readlines()
        for x in lines:
            item1 = x.replace("\n","")
            item2 = re.sub(r"[-()\"#/@;:<>{~|.?!,´`]"," ",item1)
            item3 = item2.replace('…', ' ')
            item4 = item3.replace('’', ' ')
            item5 = item4.replace('„', ' ')
            item6 = item5.replace('“', ' ')
            item7 = re.sub(r"[0-9]","",item6)
            self.nlines.append(item7.split())
#            print(self.nlines)
        return self.nlines

    def data_len(self,data, object):
        """
        Kann man immer mal verwenden... gibt die die Länge & Type der Objekte an
        """
        print(f"  Die Anzahl Wörter in {object}=", len(data))
#        print(f"Das Objekt '{object}' ist :", type(data))


    def shape_nlines(self, nlines):
        """
        Macht nlines (blöder Name) kleiner und entfernt die Stopwords
        """
        for i, item in enumerate(nlines):
            for item2 in item:
                self.wörter_org.append(item2)
                if item2.lower() not in self.stopwords_list:
                    self.clean_data.append(item2)
        return self.clean_data , self.wörter_org

    def create_blob(self):
        """
        hier kommt HanTa 
        """
        print(".")
        print('   >>>  HanoverTagger  GermaLemma  with TigerCorpus   <<<')
        tagger = ht.HanoverTagger('morphmodel_ger.pgz')
#
# Siehe: Christian Wartena (2019). A Probabilistic Morphology Model for German Lemmatization. 
# In: Proceedings of the 15th Conference on Natural Language Processing (KONVENS 2019): Long Papers. Pp. 40-49, Erlangen. )
#
        def convert(lst):
            return ' '.join(lst).split()
#
        pepe = convert(self.clean_data)
        tags = tagger.tag_sent(pepe)
#
        lemma_list = []
        for item in tags:
            lemma_list.append(item[1])
#
        self.hanta_lemma = lemma_list
#        print(lemma_list)
#
        blob_wtf = tbde(str(self.clean_data))
#        blob_wtf.words.singularize()
        self.blob_lemma = _lemmatizer.lemmatize(str(blob_wtf))
        self.blob_polarity = tbde(str(self.blob_lemma))
#        blob_wtf.parse()
        print("                      -/-                             ")
#        print("             TF-IDF  Auswertung        ")
        return self.blob_lemma, self.blob_polarity, self.hanta_lemma

    def show_polarity(self, blob_object):
        """
        Zeigt die Polarität des Datensatzes von einem blob object
        """
        print(" Polarität nach TextBlob_DE: ")
        ppol = blob_object.sentiment.polarity
        print(' Polarity:     ',format(ppol))
        print(' Subjectivity: ', blob_object.sentiment.subjectivity)
        print("-------------------------------------------------------------------")

    def tfidf_calculate(self):
        """
        Erechnet den Tfidf von in jeweils 1/3 geteilte Abschnitte des sliste_n 
        """
        self.sliste_n = [x for (x,y) in self.blob_lemma if y not in ('N')]
        def tf(word, blob):
            return blob.words.count(word) / len(blob.words)
        def n_containing(word, bloblist):
            return sum(1 for blob in bloblist if word in blob.words)
        def idf(word, bloblist):
            return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
        def tfidf(word, blob, bloblist):
            return tf(word, blob) * idf(word, bloblist)

        nb1 = int(len(self.clean_data) * 0.333)
        nb2 = nb1 * 2
        nb3 = len(self.clean_data)
        doku1= tbde(str(self.sliste_n[0:nb1]))
        doku2= tbde(str(self.sliste_n[nb1:nb2]))
        doku3= tbde(str(self.sliste_n[nb2:nb3]))

        bloblist = [doku1,doku2,doku3]
        for i, item in enumerate(bloblist):
            print("Top words in document {}".format(i + 1))
            scores = {word: tfidf(word, item, bloblist) for word in item.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for element, score in sorted_words[:3]:
                print("\tWord: {}, TF-IDF: {}".format(element, round(score, 4)))
        return self.sliste_n

    def create_dict(self, liste):
        """
        Erzeugt ein Dictionary, ein DataFrame sowie die uniques von sliste_n
        : param : liste - eine list[]
        """

        liste = self.hanta_lemma
        column = "   Text Liste   "
        self.my_dict[f'{column}'] = liste
        self.datafr = pd.DataFrame.from_dict(self.my_dict, orient='index').transpose()
        self.datafr.dropna(inplace=True)
        uniq = pd.unique(liste)
        self.unique= pd.Series(uniq)
        self.unique.dropna(inplace= True)
#        print("  ???        ")
#        print(self.datafr)
#        answer = input("Do you want to see the unique items? (y/n)")
        if True:
            self.lenn = len(uniq)
            print("        Anzahl der Unique Woerter: TK3=", len(uniq))
        print(" ")
        print("   >>>  Start Emotion Analysis Plutschik <<<  ")
        if len(uniq) == 0 :
            sys.exit('Keine Textdaten nach Entfernung Stopwords')
#PD

        for index, row in self.datafr.iterrows():
            self.labels = {} # Note: Hier hatte ich für sehr lange Zeit ein Fehler. 
            uncommons = list(set(self.unique) - set(row))
            commons = list(set(self.unique).intersection(row))
            for uc in uncommons:
                self.labels[uc] = 0
            for com in commons:
                self.labels[com] = 1
            self.encoded_vals.append(self.labels)

        print("        ----> Encoded Done <-----    ")
        print("..")
        return self.encoded_vals, self.datafr, self.my_dict, self.unique, self.lenn

    def plot_encoded(self):
        """
        Erzeugt ein DataFrame mit encoded_vals und plottet es dann
        """
        enc_df = pd.DataFrame(self.encoded_vals)
        self.df = enc_df.sum().to_frame('Häufigkeit').sort_values('Häufigkeit',ascending=False)
        print(self.df)
        self.df.to_csv(f'unique_count-{tsname}.csv',sep=';')
        self.df.sum()
        self.df.index.name = "Wort"
        self.df = self.df.reset_index()
        enc_df.sum().to_frame('Häufigkeit').sort_values('Häufigkeit',ascending=False)[:49].plot(kind='bar',figsize=(9,6),title="Verteilung der Wörter")
        plt.xlabel('Die häufigsten 50 Wörter im Text-Corpus')
        plt.ylabel('Anzahl')
        plt.xticks(rotation=90)
        plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None) 
        plt.savefig(f"Balken_{tsname}.png", bbox_inches='tight') 
        return self.df

    def emmo(self,filenrc):
        nrc_lex_df = pd.read_csv(filenrc, header=0,sep=";",encoding="utf8")
        #print(nrc_lex_df)
        self.merged_df = self.df.merge(nrc_lex_df, on="Wort")
        pd.set_option("display.max_columns",None)
        self.merged_df['Positive'] = self.merged_df['Positive']* self.merged_df['Häufigkeit']
        self.merged_df['Negative'] = self.merged_df['Negative']* self.merged_df['Häufigkeit']
        self.merged_df['Zorn'] = self.merged_df['Zorn']* self.merged_df['Häufigkeit']
        self.merged_df['Erwartung'] = self.merged_df['Erwartung']* self.merged_df['Häufigkeit']
        self.merged_df['Ekel'] = self.merged_df['Ekel']* self.merged_df['Häufigkeit']
        self.merged_df['Furcht'] = self.merged_df['Furcht']* self.merged_df['Häufigkeit']
        self.merged_df['Freude'] = self.merged_df['Freude']* self.merged_df['Häufigkeit']
        self.merged_df['Traurigkeit'] = self.merged_df['Traurigkeit']* self.merged_df['Häufigkeit']
        self.merged_df['Überraschung'] = self.merged_df['Überraschung']* self.merged_df['Häufigkeit']
        self.merged_df['Vertrauen'] = self.merged_df['Vertrauen']* self.merged_df['Häufigkeit']  
#        self.merged_df.sort_values('Häufigkeit')
        pd.set_option('display.max_columns', None)
        print(self.merged_df.head(10))
        print("-------------------- START Non-List-----------------------------------------")
        s1 = (set(self.df['Wort']))
        s2 = set(self.merged_df['Wort'])
        s3 = s1 - s2
        print(s3)
        df3 = pd.DataFrame(s3)
        print(" ")
        df3.to_csv("non_used_words.txt",sep=';')
        print("-------------------- ENDE Non- List-----------------------------------------")
        self.merged_df.dtypes
        self.merged_df.to_csv(f'mrg_8emotions_{tsname}.csv',sep=";")
        self.merged_df.loc['Sum Fruit'] = self.merged_df.sum()
        self.E_sum = self.merged_df.loc['Sum Fruit']
#        print(type(self.E_sum))
        print(".")
        print(" Häufigkeitsverteilung der 8 Emotionen:")
        print("-------------------:")
        all_emmos = self.E_sum[11] + self.E_sum[10] + self.E_sum[9] + self.E_sum[8] + self.E_sum[7] + self.E_sum[6] + self.E_sum[5] + self.E_sum[4]
#        all_emmos = self.E_sum[1]
        print((self.E_sum[1:]))
        self.ser = self.E_sum[7],self.E_sum[11],self.E_sum[8],self.E_sum[5],self.E_sum[4],self.E_sum[6],self.E_sum[9],self.E_sum[10]
#        print(self.ser)
#       Errechne Polarity aus NRC   p = [-1;+1]
        print("------------------------------------------------------------------------")
        print("    Polarität aus NRC: ")
#        print(self.E_sum[2],self.E_sum[3])
        p_sum = int(self.E_sum[2]) + int(self.E_sum[3])
        pos_p = int(self.E_sum[2])
        neu_p = int(self.E_sum[1])
        pos_px = pos_p / p_sum + 0.0000001
        neutral = self.lenn-p_sum
        if neutral < 0:
           neutral = 0
           print("    ---- TOP -----")
        print("    Summe aus pos. /  neg. Emotionen: ",p_sum,"  ....(TK4)") 
        self.TK4 = p_sum
        print("            Summe neutrale Emotionen: ",neutral)
        pol_nrc = 2.0 * pos_px - 1
        print("                                            ")
        text = colored('     Polarity  PI :', 'green', attrs=['reverse', 'blink'])
        if pol_nrc < 0 :
           text = colored('     Polarity  PI :', 'red', attrs=['reverse', 'blink'])
        lenworig = len(self.clean_data)
        emmo_index1 = p_sum/lenworig
        emmo_index2 = all_emmos/lenworig
        print("                                            ")
        print(text,round(pol_nrc,3))
        print("       Polarity NRC-PI :",round(pol_nrc,3),"  ...[-1;+1]")
        print("    Sentiment-Index SI :",round(emmo_index1,3),"  ...[0-2]")
        print("     Emotions-Index EI :",round(emmo_index2,3),"  ...[0-8]")
        self.PI = pol_nrc
        self.SI = emmo_index1
        self.EI = emmo_index2
# 
        self.recipe = ["188 Furcht",
                  "175 Vertrauen",
                  "249 Freude",
                  "301 Erwartung",
                  "221 Wut",
                  " 122 Ekel",
                  " 144 Traurigkeit",
                  "102 Überraschung"]
        print(" ") 
        print("    Donut Plot:") 
        return self.recipe,self.ser,self.TK4
#
        
#tsname = "./Corpus_NRC.txt"
#
#
#
stopwords_de = "STP_de.txt"
#
my_stop_words = ['kirchenmayer','kirchenmayers','helga','–','zeller']
#
run = Taz_Exploration(tsname, my_stop_words) 
run.load_stpwords(stopwords_de)
print("  Stop Words removed: ")
run.open_clean_file(tsname) 
#run.data_len(run.sliste, 'sliste')
#
run.shape_nlines(run.nlines)
run.data_len(run.wörter_org, 'wörter_org: TK1')
TK1 = len(run.wörter_org)
run.data_len(run.clean_data, 'clean_data: TK2')
TK2 = len(run.clean_data)
#print("           TF-IDF  Auswertung        ")
run.create_blob()
#print(" ---------------------------------------------------------------------- ")
#run.show_polarity(run.blob_polarity)

#run.tfidf_calculate()
run.create_dict(run.sliste_n)
run.plot_encoded()
#print(run.df)
emmo_lexicon = './NRC_de.csv'
#emmo_lexicon = './erg.csv'
run.emmo(emmo_lexicon)
#print(run.merged_df)
print("   ")
print("    Emotionsverteilung: ")
#print(run.recipe)
#
#
#recipe = ["288 Furcht","75 Vertrauen","249 Freude","301 Erwartung", "221 Wut"," 122 Ekel"," 144 Traurigkeit","102 Überraschung"]
vertlg = run.recipe
fig, ax = plt.subplots(figsize=(8 ,8), subplot_kw=dict(aspect="equal",anchor='SE'))
#
data = [float(x.split()[0]) for x in vertlg]
ingredients = [x.split()[-1] for x in vertlg]
data = run.ser
print(data)
ingredients = ["Furcht\n Angst",
                  "Vertrauen\n Akzeptanz",
                  " Freude\n Heiterkeit",
                  "Erwartung\n Interesse",
                  "Wut\n Ärger",
                  "Ekel\n Abscheu",
                  "Traurigkeit\n Kummer",
                  "Überraschung\n Erstaunung"]
print(" ")
print("Furcht,Vertrauen,Freude;Erwartung,Wut,Ekel,Traurigkeit,Überraschung")
#print(ingredients)
print(" ")
#
# 
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
# print(pct)
    return "{:.1f}%\n({:d} Wrd)".format(pct, absolute )
#
wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="b"))
#      
plt.setp(autotexts, size=10, weight="bold")
plt.pie(data,labels=ingredients, colors=['yellowgreen','greenyellow','yellow','orange','red','purple','navy','green'])
#
#
ax.set_title("Plutschik  8 Emotionen: ")
my_circle=plt.Circle( (0,0), 0.7, color='lightgrey')
#
p=plt.gcf()
p.gca().add_artist(my_circle)
#plt.show()
plotname = f"Donut_{tsname}.png"
plt.savefig(f"Donut_{tsname}.png", bbox_inches='tight')
#
print("  1-Zeiler Ergebnisse   ")
data_str = str(data)
data_str = data_str.replace('(', '')
data_str = data_str.replace(')', '')
#
print('FINAL',tsname,'  ',TK1,',',TK2,',',run.lenn,',',run.TK4,',', round(run.PI,2),',',round(run.SI,2),',',round(run.EI,2),',',data_str)
print("============================================ENDE ANALYSE===================================================")
print(" ")
print(" ")
print(" ")
