import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
from textblob_de import TextBlobDE as tbde
import os,math,re,sys
from textblob_de.lemmatizers import PatternParserLemmatizer
_lemmatizer = PatternParserLemmatizer()
print("....................................................................")
now = datetime.datetime.now()
date = now.strftime('%d-%m-%Y')
zeit = now.strftime('%H:%M')
#
#tsname="xxxxx"
#tsname = "TAZ_reportage.txt"
tsname = "textle.txt"
#tsname="xxxxx"
print("Auswertung Corpus   :   ",tsname)
print("Current directory is:",os.getcwd())
print("Current time is     :   ", zeit," Uhr")

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
	print("Hallo Papa ich bins Fabian")

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
        Öffnet file und macht sie fresh & clean
        """
        f = open(tsname,'r',encoding="utf8")
        lines = f.readlines()
        for x in lines:
            item1 = x.replace("\n","")
            item2 = re.sub(r"[-()\"#/@;:<>{~|.?!,´`]","",item1)
            item3 = item2.replace('…', '')
            item4 = item3.replace('’', '')
            item5 = item4.replace('„', '')
            item6 = item5.replace('“', '')
            self.nlines.append(item6.split())
        return self.nlines

    def remove_numbers(self, element):
        """
        Entfernt die Zahlen und aus irgendwelchen beschissenen Gründen joint er sie zusammen.
        """
        self.sliste = [' '.join(x for x in i if x.isalpha()) for i in element]        
        return self.sliste

    def data_len(self,data, object):
        """
        Kann man immer mal verwenden... gibt die die Länge & Type der Objekte an
        """
        print(f"Die Länge des Corpus {object}=", len(data))
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
        Erzeugt blob objekte: 1x blob lemma liste und 1x pures blob object
        """
        blob_wtf = tbde(str(self.clean_data))
        self.blob_lemma = _lemmatizer.lemmatize(str(blob_wtf))
        self.blob_polarity = tbde(str(self.blob_lemma))
        return self.blob_lemma, self.blob_polarity

    def show_polarity(self, blob_object):
        """
        Zeigt die Polarität des Datensatzes von einem blob object
        """
        print('Polarity:     ', blob_object.sentiment.polarity)
        print('Subjectivity: ', blob_object.sentiment.subjectivity)
        print("-------------------------------------------------------------------")

    def tfidf_calculate(self):
        """
        Errechnet den Tfidf von in jeweils 1/3 geteilte Abschnitte des sliste_n 
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
        print("------------------------------------------------------------")
        column = "   Text Liste   "
        self.my_dict[f'{column}'] = liste
        self.datafr = pd.DataFrame.from_dict(self.my_dict, orient='index').transpose()
        self.datafr.dropna(inplace=True)
        uniq = pd.unique(liste)
        self.unique= pd.Series(uniq)
        self.unique.dropna(inplace= True)
        print("          ")
#        print(self.datafr)
#        answer = input("Do you want to see the unique items? (y/n)")
        if True:
            print("Anzahl der Unique Woerter:", len(uniq))
        print("          ")
        if len(uniq) == 0 :
            sys.exit('Keine Textdaten nach Entfernung Stopwords')
#PD

        for index, row in self.datafr.iterrows():
            self.labels = {} # Note: Hier hatte ich für sehr lange Zeit ein verfickten drecks Fehler. Scheinbar darf labels nicht schon in der __init__ stehen
            uncommons = list(set(self.unique) - set(row))
            commons = list(set(self.unique).intersection(row))
            for uc in uncommons:
                self.labels[uc] = 0
            for com in commons:
                self.labels[com] = 1
            self.encoded_vals.append(self.labels)

        print("        ----> Encoded Done ")
        print("_______________________")
        return self.encoded_vals, self.datafr, self.my_dict, self.unique

    def plot_encoded(self):
        """
        Erzeugt ein DataFrame mit encoded_vals und plottet es dann
        """
        enc_df = pd.DataFrame(self.encoded_vals)
        self.df = enc_df.sum().to_frame('Häufigkeit').sort_values('Häufigkeit',ascending=False)
        print(self.df)
        self.df.to_csv(f'unique_count-{date}.csv',sep=';')
        self.df.sum()
        self.df.index.name = "Wort"
        self.df = self.df.reset_index()
        enc_df.sum().to_frame('Häufigkeit').sort_values('Häufigkeit',ascending=False)[:49].plot(kind='bar',figsize=(12,8),title="Verteilung der Wörter")
        plt.xlabel('Die häufigsten 50 Wörter im Text-Corpus')
        plt.ylabel('Anzahl')
        plt.xticks(rotation=90)
        plt.show()  
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
        print(self.merged_df.head(10))
        self.merged_df.to_csv(f'mrg_TAZ_{date}.txt',sep=";")
        self.merged_df.loc['Sum Fruit'] = self.merged_df.sum()
        self.E_sum = self.merged_df.loc['Sum Fruit']
        print("-------------------------------------------------------------------------------------------")
        print(" Häufigkeitsverteilung der 8 Emotionen:")
        print("-")
        print((self.E_sum[1:]))
        self.ser = self.E_sum[7],self.E_sum[11],self.E_sum[8],self.E_sum[5],self.E_sum[4],self.E_sum[6],self.E_sum[9],self.E_sum[10]
#        print(self.ser)
#       self.E_sum = ('Freude', self.merged_de['Sum_Fruit'])
#
        self.recipe = ["188 Furcht",
                  "175 Vertrauen",
                  "249 Freude",
                  "301 Erwartung",
                  "221 Wut",
                  " 122 Ekel",
                  " 144 Traurigkeit",
                  "102 Überraschung"]
        print("------------------------------------------------------------------------------------------")
        print(" Donut Plot") 
        return self.recipe,self.ser
#
        
#tsname = "TAZ_reportage.txt"
#tsname= "./Testdaten_2.txt"
#tsname="./Testdaten_stpwords.txt"
#tsname="D:\KAKA\AA\Emotionen\LEXICA_NEU\Test_NRC.txt"
stopwords_de = "STP_de.txt"
my_stop_words = ['alexander','kirchenmayer','mayer','helga','schmidt','–','schorcht','zeller','ursula']
#
run = Taz_Exploration(tsname, my_stop_words) 
run.load_stpwords(stopwords_de)
run.open_clean_file(tsname) 
run.remove_numbers(run.nlines)    
run.data_len(run.sliste, 'sliste')
run.shape_nlines(run.nlines)
run.data_len(run.wörter_org, 'wörter_org')
run.data_len(run.clean_data, 'clean_data')
print("-------------------------------------------------------------------------")
print("           TF-IDF  Auswertung        ")
run.create_blob()
print(" ------------------------------------------------------------------------ ")
run.show_polarity(run.blob_polarity)

run.tfidf_calculate()
run.create_dict(run.sliste_n)
run.plot_encoded()
#print(run.df)
emmo_lexicon = './NRC_Emotionen_clean.CSV'
run.emmo(emmo_lexicon)
#print(run.merged_df)
print("   ")
print(" Emotionsverteilung: ")
#print(run.recipe)
# #
# print("=====================================================================================================================")
#
#recipe = ["288 Furcht","75 Vertrauen","249 Freude","301 Erwartung", "221 Wut"," 122 Ekel"," 144 Traurigkeit","102 Überraschung"]
vertlg = run.recipe
fig, ax = plt.subplots(figsize=(8 ,8), subplot_kw=dict(aspect="equal",anchor='SE'))
#
data = [float(x.split()[0]) for x in vertlg]
ingredients = [x.split()[-1] for x in vertlg]
data = run.ser
print(data)
ingredients = ["Furcht",
                  "Vertrauen",
                  " Freude",
                  "Erwartung",
                  "Wut",
                  "Ekel",
                  "Traurigkeit",
                  "Überraschung"]
print(ingredients)
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

ax.set_title("Pultchik 8 Emotionen: ")
my_circle=plt.Circle( (0,0), 0.7, color='lightgrey')
#
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
#=================================================================================================================================
 
