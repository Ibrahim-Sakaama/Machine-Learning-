#import matplotlib.pyplot as pyplot

from matplotlib import pyplot

#plot() ===> pour faire le tracage
#bo ===> tracage bleu avec des pts
#pyplot.plot([1, 2, 3, 6],[3, 4, 9, 36],'bo')

#fill_between() ===> faire le replissage entre les courbes
x = range(10)
y1 = range(10)
y2 = [x**2 for x in range(10)]
pyplot.fill_between(x, y1, y2, color='yellow')

#ssavefig() ===> auvegarder l'image dans un fichier
#pyplot.savefig

#fixer la taille d'une figure
pyplot.figure(figsize=(10,10))

#shrink ===> dimetion mya3 l fleche
#pyplot.grid() ===> afficher la grille

#matrice avec matplotlib
#matshow(mat) ===> faire la courbe du matrice




pyplot.show()
