# Reconnaissance-de-Piece-de-Monnaie-

Université Paris Cité - Traitement d'image - Enseignant : Nicole Vincent 

Analyser la valeur des pièces de monnaies sur une image et déterminer la valeur totale présente sur celle-ci

## Présentation du sujet : 
Objectif : Analyser la valeur des pièces de monnaies sur une image et déterminer la valeur
totale présente sur celle-ci.
Plusieurs étapes pour l’atteindre :
1. Rassembler une base de données
2. Définir les types d’images que le système peut analyser :
->Prise de vue : prise frontale des pièces pour avoir une forme proche d’un cercle
3. Une étape de détection : Extraction des zones d'intérêts (ie. les pièces)
4. Une étape de reconnaissance : Analyse

## Outils 
C++ : Langage de programmation 
OpenCV

## Méthodes utilisées 

### Segmentation : Au niveau de la détection des ROI :
_Segmentation : Seuillage + Transformée de Hough_:

On segmente les endroits de l’images où les pièces sont visibles (ROI). On utilise un filtre
gaussien sur l’image, et on applique HoughCircle() d’OpenCV.
Afin d’obtenir un masque on utilise un filtre de Canny, findConoutrs() et drawContours().

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/8a3e3b77-d011-432a-b6c5-ea95fa6a74cd)

### Segmentation : Au niveau de la détection des ROI
-_Extraction des pièces_: 

Une fois les cercles détectés, on procède à l’extraction d’une boîte englobante
autour de ces cercles pour créer des objets “Piece” :

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/7c467d02-e321-49e2-9c13-0f7b76afc529)

### Analyse : Classification des pièces

_Classification des pièces: 3 grandes catégories_

A partir de ces objets pièces nous procédons à une classification qui est divisée en
trois catégories : pièces 1-2 euros, pièces : 1-5 centimes (rouges), et autres
(jaune)
La classification se fera à l’aide d’images de références, tirées d’internet
présentant une bonne qualité.

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/91114f74-1579-4fcf-8e0d-1e0cad8c3cf3)

_Classification des pièces 1 et 2 euros_

Pour ces pièces, une classification à base de seuillage sur la saturation des
images dans l’espace HSV suffit. On fait un seuillage OTSU sur la saturation de
notre objet “Piece”, et sur l’Image de référence, pour ensuite appliquer un XOR, et
vérifier si la moyenne du résultat est en dessous d’un seuil fixé de manière
empirique.

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/ce4b2cf0-d14c-49ff-a344-1b76f84a202f)

_Classification des pièces 1 , 2 et 5 centimes_:

Pour ces pièces, une autre classification à base de couleur se fait, mais en utilisant
l’espace couleur Lab (le canal a). On a remarqué que les pièces rouges brillaient
comparées aux pièces jaunes.
Ensuite on procède à un template matching en utilisant ORB comme descripteur

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/9cce36bc-4f14-4d91-b894-fe371349f316)

-_Classification des pièces 10, 20 et 50 centimes_

Sur les pièces de cette catégorie on procède un template matching avec les
images de référence de ces pièces en utilisant ORB comme descripteur
10

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/2c78667a-971c-46ee-a11a-ae99fd279166)

### Résultats : 

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/be3a9324-2e10-49de-9a53-e8fd031bded9)

![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/cc76ca54-8ad9-4c59-9424-789a9650ce87)

## Auteurs :

Aïssatou Signaté , Hichem Boussaid

