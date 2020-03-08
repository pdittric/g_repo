from textblob_de import TextBlobDE as tbde
from textblob_de.lemmatizers import PatternParserLemmatizer
_lemmatizer = PatternParserLemmatizer()

import spacy
import string

#textle = ' war nicht ganz richtig. Denn die eine Gesäßbacke befand sich leicht links von der Mitte und die andere Backe leicht rechts von der Mitte. Mayer kam zu dem Schluss, dass nur sein After direkt auf der Mitte der Bank saß, bzw. schwebte.  Ein Schatten fiel auf in und störte seine Kreise. Mayer blickte hoch und höher. Eine staatliche Frau stand vor ihm, und seiner Bank, auf der er mittig saß. Er beantwortete die unausgesprochene Frage in den Augen der Frau: Bitte, nehmen Sie ruhig Platz.  Danke, sagte die stattliche Frau und setzte sich rechts neben Mayer auf die Bank. Könnten Sie vielleicht etwas nach links rutschen?  Mayer löste seinen Blick von der Schönheit der Landschaft. Stattdessen betrachtete er nun nachdenklich die Dame; die rotblonden, halblangen Haare; die modische Hornbrille; das hübsche, herzförmige Gesicht; die weiße Bluse; das Dekolleté. Dann überlegte er, wie lange es gedauert hatte, hier auf dieser Bank seine Mitte zu finden. Er hatte mehrere Fixpunkte in der Landschaft, die sich unter ihm ausbreitete wie ein Ölgemälde, in seine innere Navigation, in sein Koordinatensystem, einbezogen: den Funkmast links, den Windpark rechts, natürlich den Stand der Sonne; und sein Augenmaß. Mayer besaß ein gutes Augenmaß, das ihn selten im Stich ließ. Und all diese Mühen wären mit einem Mal hinfällig wegen einer zweifellos attraktiven Frau, die aus dem Hinterhalt aufgetaucht war und ihn aus seiner Mitte vertreiben wollte? Niemals!  Nein, es tut mir leid., sagte Mayer mit fester Stimme, die keinen Zweifel an der Unabänderlichkeit seiner Entscheidung zuließ.  Sie sind nicht sehr freundlich.  Ich wurde nicht geboren, um freundlich zu sein., brummte Mayer und bohrte seinen Blick in das Gesicht der Dame.  Sie werden schon sehen, was Sie davon haben., sagte die rotblonde Dame.  Abrupt stand Mayer auf und gab die Bank frei. Für heute hatte er genug gesehen. Morgen würde er eine neue Mitte finden. Bitte, sagte er zu der Dame, lächelte verschmitzt und machte sich auf den Weg zum Parkplatz, der nur wenige Meter vom Schloss entfernt war.'
textle = ['hatte','wurde','Bäume','viele','Belastungen','gekauft','habe','brüllte','hässliche','war']
#print(l_words)
#
print(" Ausgangstext: ")
print(textle)
#blob = tbde(str(textle))
print("..............................................")
#
blob1 = _lemmatizer.lemmatize(str(textle))
#print(blob1)
print(" TextBlob  lemma ")
sliste = ()
sliste = [x for (x,y) in blob1 if y not in ('x')]
print((sliste))
print("..............................................")
print("Jetzt -- SPACY....")
print("de_core_news_sm   ")
#
nlpeter = spacy.load("de_core_news_sm")
#doc = nlpeter(str(textle))
doc = nlpeter(str(sliste))
#
spacc = []
for token in doc:
    spacc += [token.lemma_]
#
#spacc.remove(' ')
spacc = [''.join(c for c in s if c not in string.punctuation) for s in spacc]
spacc = [s for s in spacc if s]
print(spacc)   
print("    ENDE   1   ")
#print("   jetzt: de_core_news_md   ")

