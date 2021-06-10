__Autheur__ = 'LumberJack Jyss'

# Objectif 304805

# Ouvre fichier source
Base = open ('Base.html', "r")
# Ouvre le fichier de destination
Codex = open ('Codex.html', 'w')


# Lit tout ce qu'il y a dans Base et le met dans BRUTUS
BRUTUS = Base.read()

# Affiche les longueurs en caractere des deux bases avant traitement:
LongueurBase = len(BRUTUS)
print (' Avant traitement, les longueurs sont de :')
print ('Base : ', LongueurBase)

# Parcourt BRUTUS et retire le caractere '־'
BRUTUS = BRUTUS.replace('־', '')

# Parcourt BRUTUS et retire le mot 'Chapter'
BRUTUS = BRUTUS.replace('Chapter', '')

# Parcourt BRUTUS et retire les parties entre [] => \[.*\]
import re

#Parcourt BRUTUS et retire les caractères numériques
regex=re.compile('0-9')
BRUTUS=regex.sub('',BRUTUS)

regex=re.compile('\[.*\]')
BRUTUS=regex.sub('',BRUTUS)

# Parcourt BRUTUS et retire les tabulations, espaces et retours de chariot
regex=re.compile('\s')
BRUTUS=regex.sub('',BRUTUS)

# Ecrit le contenu de BRUTUS dans Codex
Codex.write(BRUTUS)

# Affiche les longueurs en caractere des deux bases apres traitement:

LongueurCodex = len(BRUTUS)

print (' Apres traitement, les longueurs sont de :')
print ('Codex : ', LongueurCodex )

# Fermeture du fichier destination
Codex.close()

# Fermeture du fichier source
Base.close()

# différence entre les deux
print("L'objectif étant de 304805 caractères, le delata après traitement est de :")
print(LongueurCodex-304805)

# Cétation de la liste SPLITUS
SPLITUS = BRUTUS.split()

#Inversion de la liste SPLITUS
#TOTO = SPLITUS.reverse()

#N=0
#N=input('N° de lettre : ')
#n=int(N)-1
#print(type(n))
#print(BRUTUS[int(n)])
print ('ET VOICI BRUTUS!!!')
print (BRUTUS)
print('ET LA LISTE')
print(SPLITUS[:])

# Retourne 'True' si au moins un des caractères d'une str est alphanumérique
print(" Est-ce qu'au moins un des carateres de BRUTUS est alnum?")
print(BRUTUS.isalnum())

#print('ET LA REVERSE')
#print(SPLITUS.reverse())
