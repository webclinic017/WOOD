__Autheur__ = 'LumberJack Jyss'

# Objectif 304805

# Ouvre fichier source
Base = open ('Base.html', "r")
# Ouvre le fichier de destination
Codex = open ('Codex.txt', 'w')
Ligne=open('Linear.txt','w')
Psoukim=open('Psoukim.txt','w')
Milot=open('Milot.txt','w')
# Lit tout ce qu'il y a dans Base et le met dans BRUTUS
BRUTUS = Base.read()
LongueurBase = len(BRUTUS)


# Parcourt BRUTUS et retire le caractere '־'
BRUTUS = BRUTUS.replace('־', ' ')

# Retire les numérique
BRUTUS=BRUTUS.replace('0','')
BRUTUS=BRUTUS.replace('1','')
BRUTUS=BRUTUS.replace('2','')
BRUTUS=BRUTUS.replace('3','')
BRUTUS=BRUTUS.replace('4','')
BRUTUS=BRUTUS.replace('5','')
BRUTUS=BRUTUS.replace('6','')
BRUTUS=BRUTUS.replace('7','')
BRUTUS=BRUTUS.replace('8','')
BRUTUS=BRUTUS.replace('9','')
BRUTUS=BRUTUS.replace('A','')
BRUTUS=BRUTUS.replace('B','')
BRUTUS=BRUTUS.replace('C','')
BRUTUS=BRUTUS.replace('D','')
BRUTUS=BRUTUS.replace('E','')
BRUTUS=BRUTUS.replace('F','')
BRUTUS=BRUTUS.replace('G','')
BRUTUS=BRUTUS.replace('H','')
BRUTUS=BRUTUS.replace('I','')
BRUTUS=BRUTUS.replace('J','')
BRUTUS=BRUTUS.replace('K','')
BRUTUS=BRUTUS.replace('L','')
BRUTUS=BRUTUS.replace('M','')
BRUTUS=BRUTUS.replace('N','')
BRUTUS=BRUTUS.replace('O','')
BRUTUS=BRUTUS.replace('P','')
BRUTUS=BRUTUS.replace('Q','')
BRUTUS=BRUTUS.replace('R','')
BRUTUS=BRUTUS.replace('S','')
BRUTUS=BRUTUS.replace('T','')
BRUTUS=BRUTUS.replace('U','')
BRUTUS=BRUTUS.replace('V','')
BRUTUS=BRUTUS.replace('W','')
BRUTUS=BRUTUS.replace('X','')
BRUTUS=BRUTUS.replace('Y','')
BRUTUS=BRUTUS.replace('Z','')
BRUTUS=BRUTUS.replace('a','')
BRUTUS=BRUTUS.replace('b','')
BRUTUS=BRUTUS.replace('c','')
BRUTUS=BRUTUS.replace('d','')
BRUTUS=BRUTUS.replace('e','')
BRUTUS=BRUTUS.replace('f','')
BRUTUS=BRUTUS.replace('g','')
BRUTUS=BRUTUS.replace('h','')
BRUTUS=BRUTUS.replace('i','')
BRUTUS=BRUTUS.replace('j','')
BRUTUS=BRUTUS.replace('k','')
BRUTUS=BRUTUS.replace('l','')
BRUTUS=BRUTUS.replace('m','')
BRUTUS=BRUTUS.replace('n','')
BRUTUS=BRUTUS.replace('o','')
BRUTUS=BRUTUS.replace('p','')
BRUTUS=BRUTUS.replace('q','')
BRUTUS=BRUTUS.replace('r','')
BRUTUS=BRUTUS.replace('s','')
BRUTUS=BRUTUS.replace('t','')
BRUTUS=BRUTUS.replace('u','')
BRUTUS=BRUTUS.replace('v','')
BRUTUS=BRUTUS.replace('w','')
BRUTUS=BRUTUS.replace('x','')
BRUTUS=BRUTUS.replace('y','')
BRUTUS=BRUTUS.replace('z','')
BRUTUS=BRUTUS.replace('*','')
BRUTUS=BRUTUS.replace(':','')
BRUTUS=BRUTUS.replace(',','')
BRUTUS=BRUTUS.replace(';','')
BRUTUS=BRUTUS.replace('(','')
BRUTUS=BRUTUS.replace(')','')
BRUTUS=BRUTUS.replace('/','')
BRUTUS=BRUTUS.replace('?','')
BRUTUS=BRUTUS.replace('.','')
BRUTUS=BRUTUS.replace('&','')
BRUTUS=BRUTUS.replace('>','')
BRUTUS=BRUTUS.replace('<','')
BRUTUS=BRUTUS.replace('!','')
BRUTUS=BRUTUS.replace("'",'')
BRUTUS=BRUTUS.replace('"','')
BRUTUS=BRUTUS.replace('׳','')
BRUTUS=BRUTUS.replace('״','')
BRUTUS=BRUTUS.replace('ֿֿֿֿֿ ','')
BRUTUS=BRUTUS.replace('_','')
BRUTUS=BRUTUS.replace('^','')
BRUTUS=BRUTUS.replace('#','')
BRUTUS=BRUTUS.replace('$','')
BRUTUS=BRUTUS.replace('%','')
# Ecrit le contenu de BRUTUS dans Codex
Codex.write(BRUTUS)

# Parcourt BRUTUS et retire les parties entre [] => \[.*\]
BRUTUS=BRUTUS.replace('[','')
BRUTUS=BRUTUS.replace(']','')
temp_milot=BRUTUS
temp_milot=temp_milot.replace('\n',' ')
milot=temp_milot.split(' ')
for n in milot:
	if n=='':
		milot.remove(n)
BRUTUS=BRUTUS.replace(' ','')

# Ecrit le contenu de BRUTUS dans Codex
Psoukim.write(BRUTUS)
BRUTUS=BRUTUS.replace('\n','')

# Ecrit le linear
Ligne.write(BRUTUS)
c=0
for n in BRUTUS:
	if n=='\\' :
		c+=1
print('Nbre itérations : ',c,'\n')


print ('ET VOICI BRUTUS!!!')
print (BRUTUS)

# Affiche les longueurs en caractere des deux bases apres traitement:

LongueurCodex = len(BRUTUS)
print (' Avant traitement, les longueurs sont de :')
print ('Base : ', LongueurBase)
print (' Apres traitement, les longueurs sont de :')
print ('Codex : ', LongueurCodex )


# Fermeture du fichier destination
Codex.close()

# Fermeture du fichier source
Base.close()

# différence entre les deux

print("L'objectif étant de 304805 caractères, le delata après traitement est de :")
print(LongueurCodex-304805)
print(type(BRUTUS))

print('Le nbre de fois : ', milot.count('יהוה'))
print(milot[13],' ',milot[26],' ',milot[45],' ',milot[52],' ',milot[63],' ',milot[72],' ',milot[91],'\n')
print(milot[14],' ',milot[27],' ',milot[46],' ',milot[53],' ',milot[64],' ',milot[73],' ',milot[92])



