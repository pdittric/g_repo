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