# Reconnaissance-de-Piece-de-Monnaie-

Université Paris Cité 2022 - Traitement d'image - Enseigant : Nicole Vincent 

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
->Segmentation : Seuillage + Transformée de Hough :
On segmente les endroits de l’images où les pièces sont visibles (ROI). On utilise un filtre
gaussien sur l’image, et on applique HoughCircle() d’OpenCV.
Afin d’obtenir un masque on utilise un filtre de Canny, findConoutrs() et drawContours().
![image](https://github.com/Calliope-commits/Reconnaissance-de-Piece-de-Monnaie-/assets/61286710/8a3e3b77-d011-432a-b6c5-ea95fa6a74cd)
