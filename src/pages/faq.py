import streamlit as st

WEEKS = 13
TITLE = "FAQs"
st.set_page_config(page_title=TITLE, page_icon="🗺️")
st.sidebar.header(TITLE)

tabs = st.tabs(list(map(lambda x: str(x), range(1, WEEKS + 1))))


tabs[0].markdown(
    """
# 6.1.1 EINFÜHRUNG  (WOCHE 1)
## Zu Kap. 0.1.3
### Wie lassen sich folgende Begriffe erläutern?

#### 1. Intrinsischen und extrinsische Kameraparameter  
Intrinsisch $\\coloneqq$ 'konstante' Eigenschaften der Kamera (Focal Length, Aspect Ratio, Verzerrungsparameter)  
Extrinsich $\\coloneqq$ Eigenschaften der Kamera relativ zur Umwelt (Position, Blickrichtung)

#### 2. Standardstereogeometrie  
Zwei Kameras, parallel zueinander, beide Bildebenen in der gleichen Ebene

#### 3. Levenberg Marquardt und Bündelblockausgleichung  
Verfahren zur Minimierung von nichtlinearen Optimierungsproblemen. Für Bündelblockausgleich: Relative Position aller Kamerastandpunkte wird mit allen anderen Standpunkten verglichen anstatt nur mit dem Ersten.

#### 4. Kamerakalibrierung nach Zhang  
Methode um Verzeichnungsparameter (radial + axial) zu bestimmen.

#### 5. Disparitätsbild ("disparity")  
Bild, in dem jeder Pixel einen Wert hat, der beschreibt, wie viele Pixel Unterschied horizontal zwischen der Projektion des selben Weltpunktes in zwei unterschiedliche Kamerabildern besteht. Benötigt Standardstereogeometrie.

#### 6. SIFT  
Scale Invariant Feature Transform  $\\implies$ Algorithmus der Markante Punkte in Bildern identifiziert. Hilfreich, um die gleichen markanten Punkte in zwei Bildern zu zuzuordnen.

#### 7. Epipolargeometrie berechnen  
Berechnung der relativen Lage zweier Bilder zueinander.

#### 8. dense 3D Modell  
3D Punktewolke, bei der jedem Pixel ein Punkt zugeordnet wird.

#### 9. sparse 3D Modell  
3D Punktewolke, bei der nur für manche Pixel ein Punkt zugeordnet wird.

#### 10. Bildrektifizierung ("rectify images")  
Entzerrung und Umrechnung von Bildern in Standardstereogeometrie.

#### 11. Triangulation ("triangulate")  
Bestimmung einer Weltkoordinate aus dem Schnittpunkt zweier Sichtstrahlen.

#### 12. Singulärwertzerlegung  
Zerlegung einer Matrix Hauptkomponenten.

#### 13. RANSAC  
Algorithmus zur Outlier Detection. Idee: Iterativ minimales Subset (für F-Matrix n = 8) an Datenpunkte nehmen, gesuchte Parameter berechnen und dann überprüfen, ob diese Parameter für einen Großteil aller Punkte plausibel sind.

#### 14. Korrespondenzsuche  
Korrespondierende Punkte zischen Bildern suchen (nach Anwendung von SIFT).

#### 15. ply und pld Datenformat  
Dateiformate zum Speichern von 3D Punktewolken.

#### 16. Pose und Homogene Matrix  
Pose $\\coloneqq$ Position+Orientierung  
Homogene Matrix $\\coloneqq$ Matrix, die die Parameter der Pose enthält.

***

## Zu Kap. 1

#### 1. Wofür steht der Begriff Lokalisation?  
Fähigkeit den eigenen Standort im Vergleich zur Umwelt zu bestimmen.

#### 2. Wofür steht der Begriff 3D Rekonstruktion?  
Erstellung eines 3D Models auf Basis von Messdaten.

#### 3. Welche Komponenten hat ein Mobile Mapping System?
 - Platform
 - Mehrere Messsensoren
"""
)


tabs[1].markdown(
    """
# 6.1.2 GEWEGUNG IM $IR^3$, KOORDINATENSYSTEME (WOCHE 2)
## Zu Kap. 2.1, Kap. 2.2, Kap. 2.3, Kap. 2.4, Kap. 2.5
#### 1. Wie ist das Skalarprodukt definiert?
Summe der paarweisen Produkte aller Komponenten zweier Vektoren.
$ \\langle u, v \\rangle \\coloneqq \\sum_{i=0}^{n}u_n v_n $

#### 2. Wie kann der Winkel zwischen zwei Vektoren berechnet werden?
Über das Skalarprodukt:
$ \\alpha = cos^{-1}(\\frac{\\langle u, v \\rangle}{||u|| ||v||})$

#### 3. Wie kann überprüft werden, ob zwei Vektoren rechtwinklig (=orthogonal) aufeinander stehen?
Wenn ihr Skalarprudukt gleich 0 ist. Siehe 2.: $ cos^{-1}(0) = 0 $

#### 4. Wie ist das Vektorprodukt definiert?
$ u \\times v \\coloneqq ||u|| ||v|| sin(\\theta)n $  
Wobei n ein Vektor im $IR^3$ ist und rechtwinklig auf u und v steht.
Es gilt auch:  
$ u \\times v = \\begin{bmatrix}
u_2 v_3 - u_3 v_2 \\\\
u_3 v_1 - u_1 v_3 \\\\
u_1 v_2 - u_2 v_1 \\\\
\\end{bmatrix} $  

#### 5. Wie ist der Zusammenhang zwischen "rechter Hand" und Vektorprodukt?  
Das Vektorprodukt berechnet aus den Vektoren u und v einen dritten Vektor w, der Rechwinklig auf u und v steht. u, v und w spannen daher ein Koordinatensystem auf.

#### 6. Wie lautet die Matrix/Vektor Form des Vektorproduktes?  
Umformen von 4.):  
$ u \\times v = \\begin{bmatrix}
 0   & -u_3 &  u_2 \\\\
 u_3 &  0   & -u_1 \\\\
-u_2 &  u_1 &  0   \\\\
\\end{bmatrix}
\\begin{bmatrix}
 v_1 \\\\
 v_2 \\\\
 v_3 \\\\
\\end{bmatrix} $  

#### 7. Welche Eigenschaften hat eine Rotation?  
Die Inverse ist gleich der Transponierten und die Determinante ist +1.

#### 8. Wie lautet der Unterschied zwischen einer Rotation und einer Spiegelung? (nicht im Skript)  
(Doch im Skript): Determinante = -1

#### 9. Gegeben sei eine 3 x3 Matrix. Wie kann überprüft werden, ob es sich um eine Rotation handelt?  
Auszug aus Praktikum:  
```python
def isRot(M):  
    # allclose handles floats more gracefully than equals  
    return (  
        np.allclose(np.matmul(M, M.T), np.identity(3))  
        and np.allclose(np.linalg.det(M), 1.0)  
    )  
```


#### 10. Wie kann eine Translation und Rotation als homogene Matrix dargestellt werden?  
In der Form $ \\begin{bmatrix}
 R & T \\\\
 0 & 1 \\\\
\\end{bmatrix} =
\\begin{bmatrix}
 r_11 & r_12 & r_13 & t_1  \\\\
 r_21 & r_22 & r_23 & t_2  \\\\
 r_31 & r_32 & r_33 & t_3  \\\\
 0    & 0    & 0    & 1    \\\\
\\end{bmatrix} $

#### 11. Welche Vorteile hat das Rechnen mit homogenen Matrizen? (nicht im Skript)  
Rotation + Translation können durch einfache Matrixmultiplikation durchegführt werden  
=> Effizienter und einfacher zu verstehen.

#### 12. Wie kann eine homogene Matrix als Koordinatensystem interpretiert werden?
Die Translation und jeweils jede Zeile der Rotation sind je ein Basisvektor.  
Das Resultierende Koordinatensystemist affin, orgthogonal und orientierungstreu.

#### 13. Wie lässt sich aus einer homogenen Matrix der Koordinatenursprung, sowie die Achsen berechnen? (nicht im Skript)  
Ursprung = $(0,0,0,0)$ ?!  
Achsen = $(T, (r_{11}-r_{13}), (r_{21}-r_{23}), (r_{31}-r_{33})$)

#### 14. Was versteht man unter einem affinen, orthogonalen und orientierungstreuen Koordinatensystem?
Affin = Grob zu verstehen als euklidisches System.  
Orthogonal = Alle Basisvektoren schneiden sich immer rechtwinklig.  
Orientierungstreu = Gegeben wenn $det(R) = 1$  

#### 15. Was versteht man unter der RPY Darstellung einer Drehung?  
Aufteilung in Roll (Kippen nach links/rechts)   
Pitch (Kippen nach vorne/hinten)   
Yaw (horizontale Dreheung nach links/rechts)  

#### 16. Wie lassen sich aus der RPY Darstellung einer Drehung die drei Winkel zurückrechnen?
Aufwändiges Gleichungssystem (tippe ich hier jetzt nicht ab...)  
Auszug aus Praktikum:  
```python
def getRPY(R):
    y = atan2(R[1, 0], R[0, 0])
    p = atan2(-R[2, 0], R[0, 0] * cos(y) + R[1, 0] * sin(y))
    r = atan2(R[2, 1] / cos(p), R[2, 2] / cos(p))
    return [r, p, y]
```


#### 17. Welche anderen Darstellungen für Drehungen gibt es noch?
- Euler Axis+Angle
- Rotation Matrix
- Quaternions
"""
)


tabs[2].markdown(
    """

# 6.1.3 KAMERAKALIBRIERUNG NACH ZHANG (WOCHE 3)

## Zu Kap. 3.1, Kap. 3.2; Kap. 3.3

#### 1. Welche Transformation ist gesucht?

#### 2. Wie lässt sich eine Kamera einfach beschreiben?

#### 3. Wofür steht der Begriff

- Lochkameramodell?

- Zentralprojektion

- tangentiale- und radiale Verzeichnung?

- externe und innere Kameraparameter?

- Bild- Kamera- und Weltkoordinatensystem?

- Brennweite?

- Bildebene (reale)

- normalisierte Bildebene

- Bildhauptpunkt

- DOF

#### 4. Wie lautet die Abbildung (inkl. Anzahl der DOF) nach dem Modell nach Zhang

- von Weltkoordinaten in das Kamerakoordinatensystem

- auf die Bildebene

- wie wird die Verzeichnung korrigiert

- in das Bildkoordinatensystem

#### 5. Wie lautet die Gesamtabbildung

#### 6. Wie kann das der Kalibrierung zugrundeliegende Fehlermodell beschrieben werden?

#### 7. Wenn lauten die praktischen Schritte zur Durchführung einer Kalibrierung?

#### 8. Was sind die inneren und externen Kameraparameter eines Stereokamerasystems (inkl. Anzahl DOF) ?

"""
)


tabs[3].markdown(
    """

# 6.1.4 STANDARDSTEREOGEOMETRIE (WOCHE 4)

## zu Kap. 4.1 (Einführung)

#### 1. In welchen Ihnen bekannten Anwendungen wird 3D Rekonstruktion durch Kameras benötigt? (Internet)

#### 2. Welches Ziel wird bei der 3D Rekonstruktion durch Kameras verfolgt?

#### 3. Wie lauten die zwei Probleme der Stereobildverarbeitung?

## Zu Kap. 4.2 (Die Standardstereogeometrie)

#### 1. Welche Modellannahmen werden an die Standardstereogeometrie gestellt?

#### 2. Wie lautet das geometrische Grundprinzip zur 3D Rekonstruktion eines Punktes (bei vorliegender Standardstereogeometrie)? Wie lassen sich die Formeln herleiten?

#### 3. Wie lässt sich der Begriff Disparität erläutern?

#### 4. Was bedeutet Basisabstand?

#### 5. Kann die Standardstereogeometrie in der Realität verwendet werden?

## Zu Kap. 4.2.5 (Konvergente Stereogeometrie)

#### 1. Wie lauten die Unterschiede zwischen Standardstereogeometrie und konvergenter Stereogeometrie hinsichtlich ihrer Modellannahmen?

#### 2. Kann die konvergente Stereogeometrie in der Realität verwendet werden?

"""
)


tabs[4].markdown(
    """

# 6.1.5 KORRESPONDENZANALYSE I – DISPARITÄTSBILD (WOCHE 5)

## zu Kap. 4.4.1, Kap. 4.4.2, Kap. 4.4.3

#### 1. Was bedeutet Korrespondenzanalyse?

#### 2. Wie lassen sich flächenbasierte Verfahren und wie lassen sich merkmalsbasierte Verfahren definieren?

#### 3. Wie lässt sich Disparitätsbild definieren?

#### 4. Welche Einschränkungen können die Suche nach Punktkorrespondenzen vereinfachen?

#### 5. Wie lauten die Schritte beim Blockmatching Algorithmus?

#### 6. Wie lässt sich der Begriff korrelationsbasiertes Matching erläutern?

#### 7. Wie lassen sich die unterschiedlichen Ähnlichkeitsmetriken beschreiben?

"""
)


tabs[5].markdown(
    """

# 6.1.6 KORRESP.‐ANALYSE II ‐ MERKMALSBASIERTE VERF. (WOCHE 6)

## zu Kap. 4.4.4

#### 1. Wie lautet der prinzipielle Ablauf und das Ziel bei der merkmalsbasierten Korrespondenzsuche?

#### 2. Wie lassen sich folgende Begriffe erläutern:

- Feature Point, Corner Point

- Feature Point Deskriptor, Keypoint Descriptor

- Feature Matching

- Point Detection, Feature Point Extraction

- Template Matching

#### 3. Wie lässt sich der Movarec Operator erläutern?

#### 4. Welche Alternativen zum Movarec Operator gibt es?

#### 5. Wie lässt sich Template Matching erläutern? (z.B. als Pseudocode)?

#### 6. Wie lauten die Schritte des SIFT Algorithmus?

- (zu Schritt 1) Wie lassen sich die Teilschritte erläutern?

- (zu Schritt 2) Wie werden die lokalen Orientierungen berechnet?

- (zu Schritt 3) Wie werden beim SIFT Algorithmus die Deskriptoren berechnet?

- (zu Schritt 4) Wie werden Deskriptoren miteinander verglichen?

"""
)


tabs[6].markdown(
    """

# 6.1.7 EPIPOLARGEOMETRIE, E MATRIX UND F MATRIX (WOCHE 7)

## zu Kap. 4.5.1, Kap. 4.5.3, Kap. 4.5.4, Kap. 4.5.5

#### 1. Wie lauten die Modellannahmen für die Epipolargeometrie?

#### 2. Wie lassen sich folgende Begriffe definieren?

- Epipolarebene?

- Epipolarlinie?

- Epipol?

#### 3. Wie lautet das epipolar constraint?

#### 4. Welche besondere Eigenschaft hat der Epipol bzgl. der Epipolarlinien?

#### 5. Wie lässt sich ein Sichtstrahl der rechten Kamera (und damit im rechten KOS beschrieben) in das Koordinatensystem der linken Kamera transformieren?

#### 6. Wie lautet die Komplanaritätsbedingung? Wie lässt sich die Komplanaritätsbedingung herleiten bzw. geometrisch erläutern?

#### 7. Welchen Zusammenhang liefert die Essentielle Matrix?

#### 8. Wie lässt sich diese aus der Komplanaritätsbedingung herleiten?

#### 9. In welchem Koordinatensystem müssen die Daten für die E Matrix vorliegen?

#### 10. In welchem Koordinatensystem müssen die Daten für die F Matrix vorliegen?

#### 11. Welche Voraussetzung wird an die Kameras in Hinblick auf die Kalibrierung beim Aufstellen der F Matrix angenommen?

#### 12. Wie wird vorgegangen, damit Bilder diese Voraussetzung erfüllen?

#### 13. Welche Struktur hat die M Matrix (=Kameramatrix)? Was ist die Aufgabe der Kameramatrix?

#### 14. Wie lautet der Zusammenhang zwischen E Matrix und F Matrix?

#### 15. Wie können Sie testen, ob es sich um eine E-Matrix oder F-Matrix oder keine von beiden handelt?

"""
)


tabs[7].markdown(
    """

# 6.1.8 WIEDERHOLUNG (WOCHE 8)

## Bewegung im $IR^3$

#### 1. Wie kann überprüft werden, ob zwei Vektoren rechtwinklig (=orthogonal) aufeinander stehen?

#### 2. Wie kann ein Vektor konstruiert werden, der rechtwinklig auf zwei Vektoren steht?

#### 3. Gegeben sei eine 3x3 Matrix. Wie kann überprüft werden, ob es sich um eine Rotation handelt?

#### 4. Wie ist der Zusammenhang zwischen einem Koordinatensystem und einer Bewegung? Das Kameramodell von Zhang

#### 1. Welche Einheiten liegen jeweils dem Kamera- Welt- Bildkoordinatensystem zugrunde?

#### 2. Wie lauten die Abbildungen von Weltkoordinaten in das Bildkoordinatensystem der Kamera nach dem Modell von Zhang

#### 3. Wie wird eine Kamera nach dem Verfahren von Zhang kalibriert?

#### 4. Was wird mit der Belichtungsautomatik eingestellt und hat das Auswirkungen auf die Kalibrierparameter? Kalibrierung eines Stereokamerasystems

#### 1. Wie viele innere Kameraparameter hat ein Stereokamerasystem?

#### 2. Wie viele externe Kameraparameter hat ein Stereokamerasystem? Die Standardstereogeometrie

#### 1. Welche Modellannahmen werden an die Standardstereogeometrie gestellt?

#### 2. Wie lautet hier das geometrische Grundprinzip zur 3D Rekonstruktion eines Punktes?

#### 3. Wie lässt sich der Begriff Disparität erläutern? Die Epipolargeometrie

#### 1. Wie lauten die Modellannahmen für die Epipolargeometrie?

#### 2. Wie lautet die Komplanaritätsbedingung?

#### 3. Wie ist der Zusammenhang zwischen F Matrix und E Matrix?

#### 4. Was sind notwendige Bedingungen für eine E Matrix bzw. F-Matrix? Korrespondenzanalyse

#### 1. Wann werden flächenbasierte Verfahren, wann merkmalsbasierte Verfahren benötigt?

#### 2. Wie lauten die Schritte beim Blockmatching Algorithmus?

#### 3. Welche Ähnlichkeitsmetriken können beim Blockmatching oder Template Matching verwendet werden?

#### 4. Wie lautet der prinzipielle Ablauf bei der merkmalsbasierten Korrespondenzsuche?

#### 5. Wie lauten die Schritte beim Template Matching Algorithmus?

#### 6. Welche Vorteile und Nachteile hat SIFT im vgl. zum Template Matching?

#### 7. in welche Teilschritte lässt sich SIFT unterteilen?

"""
)


tabs[8].markdown(
    """

# 6.1.9 8‐PUNKTE ALGORITHMUS UND RANSAC (WOCHE 9)

## zu Kap 4.6.1

#### 1. Was ist das Ziel des 8-Punkte Algorithmus?

#### 2. (Schritt 1) Gegeben seien korr. Pixel. Wie lautet der Ansatz um daraus ein Gleichungssystem zum Berechnen der Fundamentalmatrix aufzustellen?

#### 3. (Schritt 1) Welche Art von Gleichungssystem liegt dem 8 Punkte Algorithmus zugrunde? Linear/nichtlinear // Homogen/inhomogen?

#### 4. (Schritt 2) Wie schaut das Gleichungssystem in Matrix Vektor Darstellung aus?

#### 5. (Schritt 3) Wieviele korrespondierende Punkte (Pixel) werden minimal benötigt, um das Gleichungssystem zu lösen?

#### 6. (Schritt 3: Wie lässt sich das Gleichungssystem mit Hilfe des SVD lösen (und damit der Parametervektor f berechnen) und wie lautet die Lösung in Pseudocode?

#### 7. (Schritt 3) Wie lässt sich der Begriff Singulärwertzerlegung erläutern?

#### 8. (Schritt 3) Wie lässt sich der Begriff kleinster Singulärwert erläutern?

#### 9. (Schritt 4) Wie kann aus der gewonnenen 3x3 Matrix die F-Matrix F berechnet werden? Aufgrund von welcher Eigenschaft von F gilt dies?

#### 10. Wie kann der 8 Punkte Algorithmus so modifiziert werden, dass daraus direkt die E Matrix berechnet wird?

#### 11. Was sind mögliche Alternativen zum 8-Punkte Algorithmus und welche Vor- und Nachteile haben diese?

## Zu Kap. 4.6.2

#### 1. In Kombination mit welchem Algorithmus können beim 8-Punkte Algorithmus Ausreißer eliminiert werden?

#### 2. Warum sind herkömmliche statistische Verfahren zur Elimination von Ausreißern meist ungenügend?

#### 3. Wie lauten die Schritte des RANSAC Algorithmus, falls als Modellgleichung eine Geradengleichung gesucht ist?

#### 4. Wie viele korr. Punkte werden minimal benötigt, falls als Modellgleichung eine Geradengleichung gesucht ist?

#### 5. Wie lauten die Schritte des RANSAC Algorithmus, falls als Modellgleichung eine F-Matrix gesucht ist?

#### 6. Wie viele korr. Punkte werden minimal benötigt, falls als Modellgleichung eine FMatrixgesucht ist?

"""
)


tabs[9].markdown(
    """

# 6.1.10 POSE BERECHNEN UND TRIANGULATION (10. WOCHE )

## Zu Kap. 4.7.1

#### 1. Wie gut lässt sich ein 3D Modell rekonstruieren, falls die E Matrix bekannt sind?

#### 2. Wie lautet der Ansatz mit Hilfe der Singulärwertzerlegung, um die pose zu rekonstruieren?

- Welche Eigenschaften muss hierbei die Matrix Rˆ ,

- welche die Matrix ˆ S erfüllen?

#### 3. Ist die Lösung eindeutig?

#### 4. Wie wird vorgegangen um aus den vier mathematisch möglichen Lösungen die geometrisch einzig sinnvolle Möglichkeit zu bestimmen?

## Zu Kap. 4.7.2

#### 1. Wie lauten die Kernidee zur Rekonstruktion eines 3D Punktes, falls alle Kameraparameter bekannt sind? Wie heißt dieses Verfahren?

#### 2. Schritt 1:

- Wie lautet die Idee, die zum beschriebenen Gleichungssystem führt?

- Wie lässt sich das Gleichungssystem formulieren und aufstellen?

- Aus wie vielen Gleichungen besteht das Gleichungssystem? Ist es linear / nichtlinear / homogen / inhomogen / überbestimmt / nicht überbestimmt?

#### 3. Schritt 2:

- Nachdem das Gleichungssystem aus Schritt 1 aufgestellt und gelöst wurde: Wie lässt sich jetzt der gesuchte 3D Punkt berechnen?

#### 4. Wofür kann der vorgestellte Algorithmus noch verwendet werden?

"""
)


tabs[10].markdown(
    """

# 6.1.11 EINFÜHRUNG IN DAS BUNDLE ADJUSTMENT (11. WOCHE)

## Zu Kap. 4.8

1. Wie lauten die Eingabedaten für das im Skript vorgestellte Bundle Adjustment? An welcher Stelle in Ihrem workflow (z.B. im gerade zu bearbeitenden Praktikum) würde er zum Einsatz kommen?

2. Was ist gesucht?

3. Wie lässt sich also "Bundle Adjustment" allgemein formulieren?

4. Wie lassen sich folgende Begriffe definieren?

- korrespondierende Pixel

- Viewpoint, Masterviewpoint

- Transformation über Masterviewpoint

5. Warum wird die Transformation über den Masterviewpoint verwendet?

6. Wie viele Freiheitsgrade gibt es bei der Berechnung zu berücksichtigen?

7. Wie lautet das Fehlermodell?

8. Wofür wird der Levenberg-Marquardt Algorithmus benötigt?

"""
)


tabs[11].markdown(
    """

# 6.1.12 BILDREKTIFIZIERUNG (12. WOCHE)

## Zu Kap. 4.3

1. Was lässt sich unter dem Begriff Bildrektifizierung verstehen?

2. Welche Annahmen werden getroffen?

3. Wie können diese Annahmen für reale Bilder auch eingehalten werden?

4. Wie lautet die Kernidee beim Bildrektifizieren?

5. Welche Vorteile haben rektifizierte Kameras?

6. Zur Konstruktion der Rotationsmatrix: In welchen drei Schritten wird die Rotationsmatrix konstruiert?

7. Algorithmus (Rectification)

- (Schritt 1) Wie lassen sich jetzt aus der so gewonnenen Matrix die Transformationen Rl und Rr berechnen?

- (Schritt 2 und 3): Wie lassen sich jetzt so die Bildinhalte neu berechnen?

"""
)


tabs[12].markdown(
    """

# 6.1.13 WORKFLOWS UND WIEDERHOLUNG (13. WOCHE)

## Aufgabe (workflows)

## Sie wollen die Ego Motion eines Autos berechnen. Entwickeln Sie einen workflow in Pseudocode!

1. Sie wollen mit einer Spiegelreflexkamera ein 3D Modell (als Punktwolke) einer Hausfassade generieren. Entwickeln Sie einen workflow in Pseudocode!

2. Sie wollen mit einem UAV ein 3D Modell einer Kirche generieren, indem Sie mehrere GPS Koordinaten anfliegen und von dort ein Photo auslösen. Generieren Sie einen workflow in Pseudocode!

3. Sie wollen mit einem mobilen Stereokamerasystem eine 3D Punktwolke generieren (dessen Bildaufnahmefrequenz relativ hoch ist). Entwickeln Sie hierzu einen workflow in Pseudocode.

## Wiederholung

1. Rotation

2. parametrisierte Form einer Drehung. Wozu wird diese benötigt?

3. Zhang

4. E Matrix, F Matrix, Zusammenhang zwischen beiden

5. Epipolarebene, Epipolarlinie

6. Standardstereogeometrie

7. Bundle Adjustment

8. Shift, Surf

9. Feature Tracking

10. Ransac

11. Homographie

12. Epipol

13. Levenberg-Marquardt

14. density Map, SGM, Blockmatching

15. dichtes und dünnes 3D Modell

16. Arten der 3D Rekonstruktion

17. Rectify

"""
)
