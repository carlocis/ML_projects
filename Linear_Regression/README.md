# Machine learning supervisé - Régressions linéaires


<!-- TOC START min:2 max:4 link:true asterisk:true update:true -->
* [Introduction](#introduction)
  * [Qu’est ce que le machine learning supervisé ?](#quest-ce-que-le-machine-learning-supervisé-)
  * [Pourquoi faire du machine learning?](#pourquoi-faire-du-machine-learning)
* [Révision régressions linéaires](#révision-régressions-linéaires)
  * [Régressions linéaires simples](#régressions-linéaires-simples)
    * [Définition](#définition)
    * [Les hypothèses derrière une régression linéaire](#les-hypothèses-derrière-une-régression-linéaire)
    * [Estimation](#estimation)
  * [Régression linéaire multiple](#régression-linéaire-multiple)
    * [Définition](#définition-3)
    * [Les hypothèses pour une régression linéaire multiple](#les-hypothèses-pour-une-régression-linéaire-multiple)
    * [Dernières remarques](#dernières-remarques)
<!-- TOC END -->

## Introduction

### Qu’est ce que le machine learning supervisé ?

Le machine learning une science multidisciplinaire qui a pour ambition de permettre à une machine de résoudre des problèmes complexes qui ne peuvent être appréhendés par des algorithmes simples. Par exemple la prédiction du taux de conversion sur un site internet ou du prix de l'immobilier sont des problèmes qui ne peuvent être résolus que par une série de règles.

Le **machine learning supervisé** est une branche de cette discipline qui vise à résoudre des problèmes pour lesquels on dispose déjà d'exemples résolus. Par exemple on rassemble des données sur un échantillons de logements à San Francisco qui décrivent leur localisation, diverses caractéristiques et le montant du loyer. Si notre problème est d’estimer le montant du loyer d’un logement dans cette même ville qui n’est pas dans notre base, on construira à partir de nos données un modèle qui estime le montant du loyer en fonction des caractéristiques du logement et on appliquera ce modèle au logement inconnu pour en estimer le loyer. Ce problème relève de l’**apprentissage supervisé**, car au moment de construire le modèle on connaissait les valeurs prises par la **variable** que l’on souhaite estimer, qu’on appelle la **variable cible**.


### Pourquoi faire du machine learning?

Une fois qu’on a posé un problème, choisi une variable cible et rassemblé un certain nombre de variables explicatives, les objectifs de la modélisation statistique supervisée se rassemblent en trois grandes catégories non exclusives :


*   Description : on peut chercher à comprendre les relations qui peuvent exister entre la variable cible ```Y``` et les variables explicatives ```X1, ..., Xp``` afin par exemple de sélectionner celle qui sont le plus pertinentes ou obtenir une visualisation des comportements dans la population observée (attention ici, population est employé comme un terme statistique et peut aussi bien désigner des personnes, des pays, des transactions financières, etc…)
*   Explication : lorsqu’on a une connaissance du sujet traité, comme c’est souvent le cas en économie ou en biologie par exemple, l’objectif est de construire un test qui permet de vérifier ou confirmer des résultats théorique dans des situations pratiques.
*   Prédiction : ici on met l’accent sur la qualité des estimateurs et des prédicteurs, on ne cherche pas nécessairement à modéliser le mieux possible la population observée mais à construire un modèle qui permette de produire des prédictions fiables pour de futures observations.


## Révision régressions linéaires

Dans cette section nous nous intéresserons aux modèles de régression, qui rassemblent les solutions qui permettent d’estimer une variable cible numérique et **continue**. On s’intéressera en particulier à l’estimation du prix de l’immobilier à Boston, grâce à un jeu de données fourni dans le package scikit learn de python.


### Régressions linéaires simples

#### Définition

Les modèles de régressions linéaires simples sont fondés sur l’équation linéaire suivante :


![](https://drive.google.com/uc?export=view&id=10gr-X7vhbO_PdRfeB8ZZE1HVZyg_O0nX)



Ici, ```Y``` représente la variable cible, c'est à dire la variable dont on souhaite estimer la valeur, ```X``` est la variable explicative que l’on a choisi pour estimer la variables cible. ```β0, β1, ∈``` sont respectivement l’intercept (c’est à dire le niveau 0 de Y lorsque X vaut 0), le coefficient associé à X (c’est le paramètre du modèle qui mesure l’influence de X sur Y, si X augmente de 1, Y augmentera de ```β1```), et l’erreur ou résidu du modèle. En effet, l’équation ci-dessus est la représentation d’un modèle statistique : il est vrai en moyenne mais n’a pas la prétention d’être exact, ce qui explique la présence du résidu.

En fonction des individus (ou nuage de point), votre modèle va trouver la ligne qui se rapproche le plus possible tous les individus à la fois. Voici ce à quoi cela ressemble visuellement :


![](https://drive.google.com/uc?export=view&id=1TO3HA0zSs3O2AtNmJSpwuNtqkh5ASGOi)



##### Variables dépendantes

En Machine Learning, on distingue toujours les **variables dépendantes/variables cibles** des **variables indépendantes/variables explicatives.** Les variables dépendantes sont les éléments que vous cherchez à prédire. Dans l’équation du dessus, cela correspond à ```Y```.



##### Variables Indépendantes

Les variables indépendantes, représentées par ```X``` sont vos prédicteurs ou les facteurs qui vont permettre de déterminer la valeur de ```Y```. Par exemple, si nous essayons de prédire le salaire de quelqu’un en fonction de son nombre d’années d’expérience. La variable indépendante ```X``` correspond au nombre d’années d’expérience.



##### Coefficient

Le coefficient ```β1``` représente la _pente_ ou le poids qu’aura votre variable indépendante dans votre équation.



##### Constante

Enfin, La constante ```β0``` représente l’endroit où votre ligne va commencer si ```X = 0```. Dans le cas de prédiction de salaires par rapport aux années d’expérience, même à 0 années d’expériences (```X = 0```), le salaire minimum de départ est différent de 0.



##### Résidu

Le résidu, souvent noté ```∈``` correspond à l’erreur commise lors de la modélisation. Cette erreur correspond à toute l’information qui n’est pas expliquée par le modèle. On suppose souvent que l’erreur suit une loi de probabilité particulière.



#### Les hypothèses derrière une régression linéaire

Quand vous construisez un modèle de ML, vous devrez être conscient des hypothèses que vous devez respecter pour que votre modèle fonctionne bien. Dans le cas inverse, vous aurez des performances déplorables. Voici les hypothèses d’un modèle de régression linéaire simple :


##### Linéarité

La première hypothèse est simple. Il faut que vos points suivent à peu près une droite. En d’autres termes, vous devez vous assurer que votre variable dépendante suive une croissance linéaire à mesure que vos variables indépendantes augmentent.



##### Homoscédasticité

Au delà de la complexité du modèle en lui même, cela veut dire que la variance de vos points doit être relativement la même. Si vous avez une variance énorme, cela veut dire que vous avez des points très éloignés les uns des autres et qu'il sera difficile d’avoir une ligne qui soit représentative de votre dataset.



##### Normalité des variables

Les points doivent avoir une distribution normale (ou du moins à peu de choses près), ce qui est rarement le cas. Le tout est de s'en approcher en ayant une moyenne, une médiane et un mode qui ne soit pas trop éloignés.  



#### Estimation

##### Maximisation de la vraisemblance

###### Définition

La vraisemblance d’une famille de variables aléatoires est une fonction qui donne pour chaque réalisation possible de chaque ```β1 x1``` variable aléatoire de la famille la probabilité que cette combinaison de réalisations se produise. Dans le cas de la modélisation statistique, on dispose déjà de données, on connait donc déjà les réalisations de chaque variable aléatoire (c’est à dire la valeur des variables explicatives pour chaque observation). Il s’agit donc de trouver les paramètres qui permettront de rendre le plus probable possible les observations à notre disposition.  

La vraisemblance statistique est une fonction de probabilités conditionnelles (c’est une probabilité dont la loi dépend de paramètres). Soit ```X = (x1, x2, ..., xn)```, d'un vecteur de variables aléatoires et ```Θ = (θ1, θ2, ..., θk)``` de l’ensemble des paramètres dont X dépend. La vraisemblance de X s’écrit :



![](https://drive.google.com/uc?export=view&id=1ZV_IhBbbVYsop3HjpJevUn5G3XqiEvRI)


Avec



![](https://drive.google.com/uc?export=view&id=1GA2vCGlovV20nbeHZyvp4D51jSqEgpMH)



###### Estimation par maximum de vraisemblance

Une manière d’estimer les paramètres d’un modèle est de maximiser la fonction de vraisemblance correspondante. Dans le cas de la régression linéaire simple, cette fonction est:

![](https://drive.google.com/uc?export=view&id=1-Jmzh6vtAg-l03B6CITZVj6nrFPqDDAR)


On peut obtenir cette équation de la vraisemblance grâce à l’hypothèse selon laquelle l’erreur ```∈``` suit une loi normale centrée (```E(∈) = μ = 0```) d’écart-type ```σ```. On doit donc trouver le maximum (s’il existe) de cette fonction de vraisemblance. Pour ce faire on applique un logarithme de chaque côté de l’équation afin d’obtenir une somme.



![](https://drive.google.com/uc?export=view&id=1C-zq5AwiddqOj_xvEQ49j0Sx6ap_MOQ0)



On constate que l’équation dépend uniquement des paramètres du modèle : `β0, β1` et il nous reste à trouver les valeurs pour lesquelles ![](https://drive.google.com/uc?export=view&id=1FcrkL-1KD9U_sR6cMtnl_FUPX58oSJct) est maximal.


##### La méthode des moindres carrés

###### Définition

Vous vous demandez sûrement comment on sait que la ligne de notre modèle est celle qui se rapproche “Le plus” de chacun des points de notre dataset. Et bien, c’est grâce à la méthode des _moindres carrés_. Nous n’allons pas aller trop loin dans la démonstration de la formule. Ce qu’il y a à comprendre est que l’algorithme va chercher la distance minimum possible entre chaque point dans votre graphique via cette formule :

![](https://drive.google.com/uc?export=view&id=19sTq_irrBPXjWiUyGJ1z3JKXZx1f2kvz)


Dans le cas de la régression linéaire simple, l’estimation par le maximum de vraisemblance ou celle par les moindres carrés revient à trouver la valeur extrême de la même équation.

Dans cette équation, `Yi` représente chaque individu (ou point) de votre dataset alors que  
`Ŷi` représente la prédiction de votre modèle.

Après plusieurs itérations, votre algorithme est capable de trouver le nombre minimum dans cette formule et donc avoir la meilleure ligne possible qui décrit votre dataset.



###### En python

Statsmodels est un package python qui contient la plupart des modèles statistiques que nous allons utiliser dans ce cours. On l'importe grâce à la commande suivante :


```python
import statsmodels.api as sm
```


Pour optimiser les paramètres du modèle en utiliser la méthode des moindres carrés :


```python
model = sm.OLS(y, X).fit()
```


Pour faire des prédictions à partir de données :


```python
predictions = model.predict(X)
```


Obtenir un résumé des paramètres du modèle :


```python
model.summary()
```


Visualiser le modèle et les données :


```python
import matplotlib.pyplot as plt
plt.scatter(X, y,  color='black')
plt.plot(X, predictions, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```



### Régression linéaire multiple

#### Définition

##### Pluralité des variables indépendantes

La plupart du temps, vous n’aurez pas qu’un seul facteur qui va vous permettre de prédire votre variable dépendante. Par exemple, vous pouvez prédire le salaire de quelqu’un avec son nombre d’années d’expérience mais surement aussi le type de diplôme, le secteur dans lequel la personne travaille, le sexe, le pays etc.

C’est la seule différence entre la régression linéaire simple et multiple. Vous ajoutez des variables indépendantes dans l’équation.



##### Normalisation des variables

Il est essentiel de normaliser les variables explicatives avant de calculer un modèle, sinon les variables explicatives dont les amplitudes sont les plus grandes prendront naturellement plus d’importance que les autres. On doit donc transformer les variables de manière à ce que leurs valeurs soient comparables et à ce que leur moyenne soit zéro (c’est ce qu’on appelle centrer réduire).

En python on utilise :


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)
```



##### Notations mathématiques du problème

En notant ```Y``` la variable cible, ```X1, X2, ..., Xn``` les variables explicatives, ```β0, β1, ..., βp``` les paramètres du modèle et ```∈``` le vecteur des résidus, le modèle de régression linéaire multiple s’écrit :

![](https://drive.google.com/uc?export=view&id=1hwmXKpIjCFF1FVQA0XRECJNBuiV0TOYr)


On peut également écrire le problème sous forme matricielle de la manière suivante :


![](https://drive.google.com/uc?export=view&id=1gxgLKJdZxaJ_I7JvuZOUvJ8RXZV_pGLn)



Où ```Y``` est un vecteur de dimension ```n x 1```, ```X``` est une matrice de dimensions ```n x p```, ```β0``` est un vecteur de dimensions ```p x 1``` et ```∈``` est un vecteur de dimensions ```n x 1```.



##### Matrice de variance/covariance

La matrice de variance/covariance d’une collection de ```p``` variables aléatoires indexées par ```ni``` de ```1``` à ```p``` est une matrice carré de taille ```p x p``` dont les éléments sont :


![](https://drive.google.com/uc?export=view&id=10Pf7v8eP2-0mCN09OahDKptwmNg9t8qb)


Les éléments le long de la diagonale de la matrice de variance/covariance sont les variances respectives de chaque variable aléatoire. Les autres éléments sont :



![](https://drive.google.com/uc?export=view&id=10-9UYEtuYkMgaSAj1biq5Gk7BkOyCsm0)



Les covariances entre les différentes variables aléatoires.



##### Résolution matricielle par le maximum de vraisemblance

Le calcul de l’estimateur du maximum de vraisemblance est un exercice classique en statistiques, nous présentons donc le calcul ici pour celles et ceux qui sont familiers ou souhaiteraient se familiariser avec le calcul matriciel.


![](https://drive.google.com/uc?export=view&id=1EOxcItDlclvdtBGjBHLM9w_D8U05BRGx)





Selon les hypothèses nécessaires pour pouvoir utiliser un modèle linéaire multiple, ```∈``` suit une loi Normale centrée (```E(∈) = 0n```) et de matrice de covariance diagonale ```Σ = Diag(σ1², σ2²,, ... , σp²)```. Ce qui nous amène à l’équation suivante :


![](https://drive.google.com/uc?export=view&id=1XpKB4oqWhCxKWDnUU3RT50XLrpn_hHeG)


`Σ` est une matrice diagonale, d’où

![](https://drive.google.com/uc?export=view&id=1FhvDHstPYxZ8xEA22Ng99emcRihvZ-Jc)


On applique le logarithme qui est croissant et ne change donc pas le problème d’optimisation considéré :

![](https://drive.google.com/uc?export=view&id=1fvnaiXHxSTnW8-al-cLOoKQoU7O9LmFt)



On cherche à trouver la valeur de ```β``` qui maximise l’équation ci-dessus, ce qui revient à trouver le minimum de la valeur suivante :


![](https://drive.google.com/uc?export=view&id=1xxmj3ZqhcHlw8Qrlrwwf0v4cbCdPr1-D)


![](https://drive.google.com/uc?export=view&id=1gYDJFKjndp15_fg-SYaqrmi4OGKNtGjB) est un scalaire (c’est à dire un nombre réel de dimension 1), il est donc égal à son transposé ![](https://drive.google.com/uc?export=view&id=14AslMdOhdDBJw7GqkyCbGal4K0RUXp6n), d’où :

![](https://drive.google.com/uc?export=view&id=1aFosYRSEXTh_FeyI9_jIU0m1_pV41Vsa)



On dérive par rapport à ```β``` et on obtient :

![](https://drive.google.com/uc?export=view&id=13gNeq6GQH0rI3qPKHlsijn3r6vbJkHib)



Cette solution n’est bien définie que si ![](https://drive.google.com/uc?export=view&id=1XeOultCi206kR3brpGiJBYQnvA4_d9Qy) est une matrice inversible, ce qui est vrai si les variables explicatives ne sont pas colinéaires et si ```p < n``` (le nombre de variables explicatives est inférieur au nombre d’observations).

En python, le calcul du modèle se fait grâce aux lignes de code suivantes :


```python
model = sm.OLS(y, X).fit()
```


C’est effectivement la même commande que pour la régression linéaire simple (car c’est le même modèle!) la seule chose qui change est la forme de X!



#### Les hypothèses pour une régression linéaire multiple

##### Tout ce qu’il y a dans la régression linéaire simple plus NON colinéarité

Comme vous l’imaginez déjà, les régressions linéaires multiples vont suivre les mêmes hypothèses que les régressions linéaires simples car après tout, on ajoute simplement un peu de complexité. La seule chose que vous devez ajouter dans les hypothèses est la _non-colinéarité_ des variables indépendantes.

Prenons un exemple, si vous essayez de prédire le salaire de quelqu’un en fonction de son âge et de son nombre d’années d’expérience, vous allez rencontrer un problème. En effet, entre l’âge et le nombre d’années d’expérience, il est tout à fait possible d’établir une relation entre les deux puisque, logiquement, plus vous êtes âgé, plus vous avez d’années d’expérience.

Si vous avez une colinéarité dans votre modèle, celui-ci va être biaisé et inutilisable car on ne pourra pas savoir quelle variable influence vraiment votre variable dépendante.   



##### Variables factices (_Dummy Variables_)
###### Rappel : Variables catégoriques

Pour comprendre ce que sont des variables factices, rappelons d’abord ce que sont des variables catégoriques : ce sont simplement des données qualitatives. Pensez par exemple à des pays, des tailles de chaussures etc. Vous pourriez techniquement créer une catégorie pour chaque variable.


###### Encoder des variables catégoriques

Dans des régressions, vous ne pouvez pas avoir de données textes comme variables, seuls les nombres sont acceptés. C’est pourquoi on encode les variables catégoriques et les remplace par 0 ou 1



###### Le piège des variables factices

Une fois que vous avez encodé vos variables factices, vous n’allez pas toutes les ajouter dans votre équation car vous aurez un problème de colinéarité entre votre dernière dummy variable et votre première : l’une sera l’opposée de l’autre. Vous ajouterez donc toutes les dummy variables et vous en **enlèverez 1 dans votre équation.**

En effet, une des hypothèses du modèle de régression linéaire multiple est la non colinéarité des variables explicatives : il ne doit pas exister de combinaison linéaire qui lie les variables entre elles.

Une combinaison linéaire entre des variables ```X₁, X₂, ..., Xm ``` est une somme de la forme :

![](https://drive.google.com/uc?export=view&id=1fQEaap-eKMZZXX8YBqk-brZM4ZX20Zmm)


Où les coefficients ```a₀, a₁, ..., am ``` sont des nombres réels. Les variables sont dites linéairement liées, ou colinéaires, si il existe un groupe de coefficients tels que :


![](https://drive.google.com/uc?export=view&id=1wXv8rydithC1PNtp8vOlmUUlpsfJX1i6)



Dans le cas des variables factices, la somme de toutes les variables vaut toujours 1, car chaque observation appartient nécessairement à une des modalités, de fait :


![](https://drive.google.com/uc?export=view&id=1MO57QcteGdNgMZBOnOP5NJTwipbISWK6)



Et les variables sont colinéaires. Il est donc essentiel de supprimer une des variables factices afin d’éviter la colinéarité des variables, la variable retirée correspondra à la modalité par défaut, et les coefficients associés aux autres variables représenteront l’influence de chaque modalité par rapport au niveau par défaut.

Pour réaliser simplement l’encodage des variables catégorielles on peut utiliser les commandes suivantes :


```python
X.get_dummies()
```




##### Sélection de variables et choix de modèle

La caractéristique principale de la régression linéaire multiple est qu’elle fait appel à plusieurs variables explicatives conjointement. Dès lors la question qui naturellement se pose est : quelles variables dois-je utiliser pour construire le meilleur modèle possible en fonction de mes objectifs? Cette question nous amène à introduire des critères d’évaluation de modèles et des méthodes de sélection de variables.



##### Evaluation des modèles de régression linéaire multiple

Certains des critères d’évaluation présentés ci-dessous pourront être utilisés pour d’autres modèles que la régression linéaire multiple. Il est donc d’autant plus important de les introduire dès maintenant et de bien retenir leurs interprétations respectives.



###### L’analyse de la variance (ANOVA pour Analysis Of Variance)

L’analyse de la variance permet de quantifier les performances d’un modèle statistique en terme d’erreur d’estimation. Les différentes valeurs dont nous allons parler maintenant vont être utilisées pour construire d’autres indicateurs :



*   SST : Sum of Square Total est un indicateur de la dispersion des valeurs de la variable cible ```Y``` (dont les valeurs sont notées ```y1, ... yn```) sur la population considérée, ce qui s’écrit mathématiquement :

![](https://drive.google.com/uc?export=view&id=1R_jf_Ew6Ik4ChOQSQaHFvBHuEz2Wxuvq)



C’est la somme des écarts à la moyenne au carré des valeurs prises par la variable cible ```Y``` pour les ```n``` observations considérées.

*   SSE : Sum of Square Explained est un indicateur qui représente la quantité de dispersion de la variable cible qui est expliquée par le modèle, ce qui s’écrit :

![](https://drive.google.com/uc?export=view&id=1iunTR9tQBm-cjGXjWK6aDzfFhRx7J0E2)


C’est la somme des écarts à la moyenne au carré entre les estimations du modèle pour chaque observation et la moyenne de la variable cible pour la population considérée.

*   SSR : Sum of Squared Residual est un indicateur qui quantifie l’erreur commise par le modèle, ou en d’autres termes la portion de la dispersion de la variable cible qui n’est pas expliquée par le modèle, d’où l’idée de résidu. Sa formule est la suivante :

![](https://drive.google.com/uc?export=view&id=1IbMAvJ234UTzBwtmR5UxW7bMN3nyaFPu)



Il est essentiel de bien comprendre ces valeurs car elles vont nous permettre de construire toutes les métriques d’évaluation des modèles de régression linéaire multiple que nous allons voir maintenant.

Pour résumer, SST est la variance totale de la variable cible, qui peut se décomposer en deux composantes : SSE la variance expliquée par le modèle, qui correspond à la quantité de variance de nos estimation par rapport à la moyenne réelle de la population observée et SSR qui est la somme des carrés des écarts entre nos estimations est les valeurs réelles de la variables cible. En d’autres termes SST est la quantité d’information totale, SSE est l’information expliquée par le modèle et SSR l’information qui reste à expliquer, ou l’erreur commise.



###### F-Statistique de Fisher

Un test statistique est un procédé par lequel on cherche à montrer si une hypothèse est confirmée ou infirmée par les données à notre disposition. Cette hypothèse de test, encore appelée hypothèse nulle et notée ```H₀``` se traduirait par des conséquences sur les propriétés des données observées si elle est effectivement vérifiée. Ces propriétés sont résumées par une statistique de test dont la valeur donne une idée de la probabilité qu’a ```H₀``` d’être vraie.

La F-statistique de Fisher permet de tester la véracité des hypothèses suivantes :


*   Dans le cas où l’on applique le test de Fisher au modèle dans sa globalité, l’hypothèse nulle, notée ```H₀```, est “les variables choisies pour construire le modèle ne sont pas conjointement significatives pour décrire la variable cible”. Si l’hypothèse est vraie, la F-statistique devrait suivre une loi de probabilité de distribution F de paramètres ```(n - 1, n - 1)``` où ```n```  est le nombre d'observations utilisées pour construire le modèle. Or si la valeur de la F-statistique, notée ```F```  se trouve en dehors des régions les plus probables de la distribution, on peut alors rejeter l’hypothèse nulle et conclure que le modèle choisi a un réel pouvoir explicatif sur la variable cible.

Mathématiquement, la F-statistique s’écrit :
![](https://drive.google.com/uc?export=view&id=1_3UXa1tI2vHV8hFpuLcDP_m_lJCIyvjH)



Le F-test peut également comparer deux modèles emboités (le modèle 1 qui inclus ```d``` variables explicatives et le modèle 2 qui inclus ```d'``` variables explicatives dont les ```d``` variables du modèle 1 et ```d < d'```. Dans ce cas la F-statistique suit une loi F de paramètres ```(n - 1, n - 1)``` si l’hypothèse selon laquelle le modèle le plus simple (modèle 2) parmi les deux modèles décrit mieux la variable cible est vérifiée. La formule mathématique de F est alors :


![](https://drive.google.com/uc?export=view&id=14rAVjTJT_Qe5_Q5bgeuD8yTPB9gDEUiL)



Si la valeur de F se situe dans une région peu probable de la F-distribution qu’elle est censée suivre, alors on rejette l'hypothèse et le test suggère que le modèle 2, plus complexe, apporte une information supplémentaire significative par rapport au modèle 1, plus simple.


Graphiquement le F-test peut s’illustrer ainsi :


![](https://drive.google.com/uc?export=view&id=1KKDqtiiimdfHZK59JkZ2wyvDOQUJA8ER)




En noir, nous représentons la densité de la loi F, comme dans tout test on défini un niveau ```α``` compris entre 0 et 1 qui va influencer la taille de la zone de rejet de l’hypothèse. Très souvent, on prend ```α = 5%``` lorsqu’aucune connaissance métier ne peut nous aider à moduler nos exigences de précision. Le test F est unilatéral, seules de grandes valeurs de F permettront de rejeter l’hypothèse. Plus précisément si la valeur de F se situe dans la partie supérieure de la distribution attendue équivalent à 5% de probabilité, alors on peut dire que l'hypothèse est rejetée à ```1 - α``` 95%.


Cette première métrique permet de tester l’hypothèse selon laquelle les variables explicatives n’ont pas d’influence sur la variable cible, nous allons maintenant nous intéresser à des métriques qui indiquent les performances du modèle.



* `R²`(R carré)


R² qu’on appelle en anglais R-squared, est une statistique qui quantifie le pouvoir explicatif du modèle par rapport à la variable cible.


![](https://drive.google.com/uc?export=view&id=1sXuFVB-7Tn1fi5Dr0msu7CA3Vbik2jR2)


R² est monotone croissant avec le nombre de variables explicatives qu’on ajoute au modèle. Il varie entre 0 et 1, si le modèle est peu pertinent, la somme des carrés résiduels ```SSR```  sera proche de la somme des carrées totaux ```SST```  et ```R²``` sera plus proche de 0, au contraire, si le modèle permet d’expliquer fidèlement la variable cible, alors ```SSR``` sera plus proche de 0 et ```R²```  sera plus proche de 1. Ainsi mécaniquement à chaque ajout de variable au modèle, la prédiction de ```Y``` , la variable cible, sera meilleur et ```R²```  sera plus élevé. De fait,```R²``` est un indicateur de performance qui permet uniquement de comparer deux modèles qui ont le même nombre de variables explicatives.



*  R² - ajusté


`R² - ajusté` est une version modifiée de ```R²``` qui pénalise le nombre de variables explicatives sélectionnées pour construire le modèle. Sa formule mathématique est :


![](https://drive.google.com/uc?export=view&id=1IsFth3VRK45KqnLrZLJDCE4_9IQnEv7d)


Où ```R²``` est le nombre de variables explicatives utilisées et ```n``` le nombres d’observations utilisées. La croissance de ```R²``` en fonction de ```p``` est compensée par la décroissance de ```(n-1) / (n-p-1)``` en fonction de ```p```. En conséquence, si l’apport d’information d’une variable explicative n’est pas assez important, alors ```R² - ajusté``` va décroître. De fait il est possible d’utiliser cet indicateur pour comparer entre eux les performances de modèles qui n’ont pas nécessairement le même nombre de variables explicatives.


Afin d’observer tous ces indicateurs dans python il faut utiliser la commande suivante :


```python
model.summary()
```


Notez bien que la commande .summary() n’est disponible dans le package statsmodels mais pas dans le package sklearn qui est un package plus orienté machine learning et pas vraiment statistiques et analyse de données.



###### Sélection de modèle.

Lorsqu’on a à notre disposition ```p```variables explicatives, le nombre de modèles qu’il est possible de construire peut être décompté de la manière suivante : on considère les variables une par une et on se demande à chaque fois si on la sélectionne pour le modèle ou non, de fait si on a une variable on peut construire deux modèles, celui avec ```X1``` ou celui sans ```X1```. Si on ajoute plus de variables explicatives possibles, on construit comme un arbre, les deux premières branches correspondant au fait de sélectionner ou non la variable  ```X1```. Elles se divisent elles-même en deux branches pour sélectionner ou non  ```X2```  et ainsi de suite. De fait, si on a à notre disposition ```p``` variables explicatives, on peut potentiellement construire ```2^p``` modèles avec. On ne peut pas en pratique, lorsque le nombre de variables explicatives ```p``` est grand, explorer les ```2^p``` modèles qu’il est possible de construire à l’aide de ces variables afin de sélectionner le meilleur. Différentes méthodes existent qui permettent d’éviter de faire appelle à la force brute.



        1. Pas à pas

La sélection pas à pas s’articule en trois variantes :

*   Sélection (forward) : on ajoute une à une les variables au modèle en sélectionnant à chaque pas celle dont la p-value (test de Fisher qui compare deux modèles emboîtés) est la plus faible. On arrête lorsque toutes les variables sont utilisées ou si la p-value minimale devient supérieure à une valeur seuil, qui par défaut est fixée à 0.5.
*   Elimination (backward) : On démarre cette fois-ci avec un modèle utilisant toutes les variables explicatives. A chaque étape, la variable associée à la plus grande p-value associée au test de Fisher est éliminée du modèle. La procédure s’arrête lorsque toutes variables restantes présentent des p-value supérieures à un seuil fixé par défaut à 0.1 (mais qu’on peut adapter en fonction des besoins en précision du problème considéré).
*   Mixte (stepwise) : Cet algorithme alterne entre une étape de sélection et une étape d’élimination après chaque ajout de variable, afin de retirer d’éventuelles variables qui seraient devenues moins pertinentes en présence de celles qui ont été ajoutées.



        2. Par échange

Cette méthode a pour but de trouver le meilleur modèle pour chaque niveau (pour chaque nombre de variables explicatives indépendantes) en commençant par le niveau 1. A chaque niveau, on sélectionne une variable non-encore incluse dans le modèle qui maximise l’accroissement de ```R²```. Puis, il échange tour à tour une variable présente dans le modèle avec une variable extérieure au modèle et conserve la configuration qui maximise ```R²```.

Le même algorithme peut être légèrement modifiée pour sélectionner l’échange des deux variables qui réalise le plus petit accroissement de ```R²```, l’idée derrière cette variante est d’explorer plus de modèles différents et ainsi d’atteindre un meilleur optimum.



        3. De manière globale

L’algorithme de Furnival et Wilson permet de comparer tous les modèles possibles avec pour objectif l’optimisation d’un des critères d’évaluation parmi ```R²``` et ```R² - ajusté```.



### Dernières remarques

Les modèles linéaires sont très sensibles aux valeurs extrêmes qui peuvent être présentes dans un jeu de données, le pré-traitement de votre base d’apprentissage est donc essentiel afin de ne pas voir vos résultats complètement faussés.

Les méthodes d’évaluation et de sélection de modèles introduites ci-dessus sont parfaitement valides pour l’ensemble des modèles linéaires, ainsi que la régression logistique que nous aborderons par la suite.
