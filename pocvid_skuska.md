1. Otazka: Popíšte proces získavania digitálneho obrazu: model dierkovej kamery, transformácie z 3D do 2D, čo je obrazová funkcia, aký je rozdiel medzi spojitou a diskrétnou (digitálnou) obrazovou funkciou, vzorkovanie, kvantizácia, aliasing a súvis so Shanonovou teorémou, získanie farebného obrazu a demozaikovanie, prevod farebného obrazu na šedotónový.

1. Model dierkovej kamery (pinhole camera)
- Najjednoduchší model zobrazenia 3D scény na 2D plochu
- Svetlo prechádza malým otvorom (dierkou) a premieta sa na zadnú stenu
- Matematický model:
```
x' = f * (X/Z)
y' = f * (Y/Z)
```
kde (X,Y,Z) sú súradnice 3D bodu a (x',y') sú súradnice jeho projekcie na obrazovú rovinu, f je ohnisková vzdialenosť

2. Transformácia z 3D do 2D
- Perspektívna projekcia: objekty ďalej od kamery sa javia menšie
- Homogénne súradnice: používajú sa na zjednodušenie výpočtov
- Projekčná matica:
```
[f 0 0  0]
[0 f 0  0]
[0 0 1  0]
```

3. Obrazová funkcia
- Spojitá obrazová funkcia f(x,y): 
  - Definuje intenzitu/jas v každom bode obrazu
  - (x,y) sú priestorové súradnice
  - Hodnota funkcie predstavuje jas alebo intenzitu v danom bode
  
- Diskrétna obrazová funkcia f[m,n]:
  - Vzniká vzorkovaním spojitej funkcie
  - [m,n] sú diskrétne súradnice (celé čísla)
  - Hodnoty sú kvantované do konečného počtu úrovní

4. Vzorkovanie a kvantizácia
- Vzorkovanie:
  - Prevod spojitého signálu na diskrétny
  - Vzorkovacia frekvencia musí spĺňať Shannonov teorém
  - Príklad vzorkovania:
```
Pre obrazový signál s maximálnou frekvenciou fmax = 100 Hz
Potrebná vzorkovacia frekvencia: fs > 2 * fmax = 200 Hz
```

- Kvantizácia:
  - Prevod spojitých hodnôt na diskrétne úrovne
  - Typicky 8 bitov = 256 úrovní pre šedotónový obraz
  - Príklad:
```
Hodnota 127.3 -> 127
Hodnota 127.7 -> 128
```

5. Aliasing a Shannonov teorém
- Aliasing:
  - Vzniká pri nedostatočnej vzorkovacej frekvencii
  - Prejavuje sa ako "schodovitý efekt" na hranách
  - Riešenie: anti-aliasing filter pred vzorkovaním

- Shannonov teorém:
  - fs > 2 * fmax
  - Vzorkovacia frekvencia musí byť aspoň 2-krát vyššia ako najvyššia frekvencia v signáli

6. Získanie farebného obrazu a demozaikovanie
- Bayerova maska:
  - RGGB vzor filtrov na senzore
  - 50% zelených, 25% červených, 25% modrých pixelov
  - Príklad rozloženia:
```
R G R G
G B G B
R G R G
G B G B
```

- Demozaikovanie:
  - Interpolácia chýbajúcich farebných hodnôt
  - Metódy: bilineárna interpolácia, adaptívne metódy
  - Príklad bilineárnej interpolácie pre červený kanál v zelenom pixeli:
```
R = (R1 + R2) / 2
kde R1, R2 sú susedné červené pixely
```

7. Prevod farebného obrazu na šedotónový
- Vážený súčet RGB kanálov:
```
Y = 0.299*R + 0.587*G + 0.114*B
```
- Váhy zodpovedajú citlivosti ľudského oka na jednotlivé farby

Praktické príklady:
1. Výpočet rozlíšenia potrebného pre zachytenie detailov:
```
Ak chceme zachytiť detaily veľkosti 1mm z vzdialenosti 1m:
- Použijeme vztah: rozlíšenie = (2 * π * vzdialenosť) / veľkosť_detailu
- rozlíšenie = (2 * π * 1000mm) / 1mm ≈ 6283 pixelov
```

2. Výpočet veľkosti súboru:
```
Pre 8-megapixelový farebný obraz (RGB):
- 8,000,000 pixelov * 3 kanály * 8 bitov = 192,000,000 bitov
- 192,000,000 / 8 = 24,000,000 bajtov = približne 24 MB
```

Táto látka je základom pre pochopenie digitálneho spracovania obrazu. Pri štúdiu sa zamerajte najmä na:
- Pochopenie vzťahu medzi spojitým a diskrétnym obrazom
- Význam vzorkovacej frekvencie a Shannonovho teorému
- Princípy farebného snímania a spracovania obrazu

2. Otazka: Fundamentálna matica kamery, jej rozklad na vonkajšiu a vnútornú maticu, čo popisujú jednotlivé matice, princíp procesu kalibrácie kamery, čo je reprojekčná chyba, skreslenie - radiálne a tangeciálne

Pripravím pre vás podrobný popis fundamentálnej matice kamery a procesu kalibrácie.

1. Fundamentálna matica kamery (P)
- Je 3×4 projekčná matica, ktorá transformuje 3D body na 2D obrazové body
- Zahŕňa všetky parametre kamery (vnútorné aj vonkajšie)
- Matematický zápis:
```
P = K[R|t]
kde:
P - fundamentálna matica (3×4)
K - vnútorná matica (3×3)
[R|t] - vonkajšia matica (3×4)
```

2. Rozklad na matice:

a) Vnútorná matica (K):
```
K = [fx  s  cx]
    [0   fy cy]
    [0   0   1]

kde:
fx, fy - ohniskové vzdialenosti v pixeloch
cx, cy - súradnice hlavného bodu (principal point)
s - skew parameter (väčšinou 0)
```
- Popisuje vnútorné parametre kamery
- Transformuje normalizované súradnice na pixelové súradnice
- Nezávisí od polohy kamery v priestore

b) Vonkajšia matica [R|t]:
```
[R|t] = [r11 r12 r13 t1]
        [r21 r22 r23 t2]
        [r31 r32 r33 t3]

kde:
R - rotačná matica 3×3
t - translačný vektor 3×1
```
- Popisuje polohu a orientáciu kamery v priestore
- R - rotácia kamery
- t - posun kamery vzhľadom na svetový súradnicový systém

3. Kalibrácia kamery:

a) Proces kalibrácie:
1. Získanie kalibračných snímok:
   - Použitie kalibračného vzoru (šachovnica)
   - Zachytenie vzoru z rôznych uhlov a vzdialeností

2. Detekcia význačných bodov:
   - Nájdenie rohov šachovnice
   - Určenie ich presnej pozície v pixeloch

3. Riešenie sústavy rovníc:
```
Pre každý bod:
x' = PX
kde:
x' - 2D bod v obraze
X - 3D bod v priestore
P - hľadaná fundamentálna matica
```

4. Reprojekčná chyba:
- Miera presnosti kalibrácie
- Vypočíta sa ako:
```
E = Σ ||xi' - P*Xi||²
kde:
xi' - skutočná pozícia bodu v obraze
P*Xi - prepočítaná pozícia 3D bodu do obrazu
```
- Čím menšia chyba, tým presnejšia kalibrácia
- Typické hodnoty: 0.1 - 1.0 pixela

5. Skreslenie obrazu:

a) Radiálne skreslenie:
- Spôsobuje "súdkovitosť" alebo "poduškovitosť" obrazu
- Modeluje sa polynómom:
```
x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶)
y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶)

kde:
r² = x² + y²
k1, k2, k3 - koeficienty radiálneho skreslenia
```

b) Tangenciálne skreslenie:
- Vzniká pri nedokonalom zarovnaní šošoviek
- Matematický model:
```
x_distorted = x + [2p1xy + p2(r² + 2x²)]
y_distorted = y + [p1(r² + 2y²) + 2p2xy]

kde:
p1, p2 - koeficienty tangenciálneho skreslenia
```

Praktický príklad kalibrácie:
```python
# OpenCV príklad kalibrácie
import cv2
import numpy as np

# Príprava bodov šachovnice
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Zber bodov z viacerých snímok
objpoints = [] # 3D body v reálnom priestore
imgpoints = [] # 2D body v obraze

# Pre každú kalibračnú snímku
ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

# Kalibrácia
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
```

Dôležité body na zapamätanie:
1. Fundamentálna matica P kombinuje vnútorné a vonkajšie parametre kamery
2. Vnútorná matica K popisuje vlastnosti samotnej kamery
3. Vonkajšia matica [R|t] popisuje polohu kamery v priestore
4. Kalibrácia je proces určenia všetkých týchto parametrov
5. Reprojekčná chyba slúži na overenie kvality kalibrácie
6. Skreslenie je potrebné kompenzovať pre presné merania

3. OTAZKA: Predspracovanie obrazu, na čo slúži. Čo je to šum, ako vzniká, biely, gaussovský, aditívny, multplikatívny, skladanie signálov. Histogram jasu, čo to je, ako sa počíta pre digitálnu obrazovú funkciu, čo vyjadruje pre spojitú, Bodové jasové transformácie: vyrovnávanie histogramu, ako sa vykonáva a aký má vplyv na výsledný obraz. Ako sa vykonáva jasová korekcia. Čo je to jasová stupnica, ako ju možno modifikovať. Čo je to kontrast v obraze, zvýšenie a zníženie kontrastu, gamma korekcia

Pripravím komplexnú odpoveď o predspracovaní obrazu a súvisiacich konceptoch.

1. Predspracovanie obrazu
- Účel: zlepšenie kvality obrazu pre ďalšie spracovanie
- Hlavné ciele:
  - Odstránenie šumu
  - Zlepšenie kontrastu
  - Korekcia jasových nerovnomerností
  - Zvýraznenie charakteristík pre ďalšiu analýzu

2. Šum v obraze

a) Typy šumu:
- Biely šum
  - Náhodný signál s rovnomerným výkonovým spektrom
  - Príklad v praxi:
  ```python
  white_noise = np.random.normal(0, 1, image.shape)
  noisy_image = image + white_noise
  ```

- Gaussovský šum
  - Hodnoty šumu majú Gaussovo (normálne) rozdelenie
  - Matematický model:
  ```
  p(z) = (1/√(2πσ²)) * e^(-(z-μ)²/2σ²)
  kde:
  μ - stredná hodnota
  σ² - rozptyl
  ```

b) Spôsob pridania šumu:
- Aditívny šum
  ```
  g(x,y) = f(x,y) + n(x,y)
  kde:
  f(x,y) - pôvodný obraz
  n(x,y) - šum
  g(x,y) - výsledný obraz
  ```

- Multiplikatívny šum
  ```
  g(x,y) = f(x,y) * n(x,y)
  ```

3. Histogram jasu

a) Definícia:
- Pre diskrétny obraz:
  ```
  h(rk) = nk
  kde:
  rk - k-tá úroveň jasu
  nk - počet pixelov s jasom rk
  ```

- Pre spojitý obraz:
  - Vyjadruje pravdepodobnosť výskytu danej jasovej úrovne
  - Integrál histogramu = 1

b) Výpočet histogramu v praxi:
```python
def calculate_histogram(image):
    histogram = np.zeros(256)
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1
    return histogram
```

4. Bodové jasové transformácie

a) Vyrovnávanie histogramu:
- Cieľ: rovnomerné rozloženie jasových hodnôt
- Postup:
```python
def equalize_histogram(image):
    # Výpočet kumulatívneho histogramu
    hist = np.bincount(image.ravel(), minlength=256)
    cum_hist = np.cumsum(hist)
    
    # Normalizácia
    cum_hist = (cum_hist * 255) / cum_hist[-1]
    
    # Mapovanie hodnôt
    return cum_hist[image]
```

b) Jasová korekcia:
- Lineárna transformácia:
  ```
  g(x,y) = a * f(x,y) + b
  kde:
  a - zmena kontrastu
  b - zmena jasu
  ```

5. Jasová stupnica a jej modifikácie

a) Základné transformácie:
- Lineárna:
  ```
  g = a*f + b
  ```
- Logaritmická:
  ```
  g = c * log(1 + f)
  ```
- Exponenciálna:
  ```
  g = c * (e^f - 1)
  ```

6. Kontrast v obraze

a) Definícia:
- Rozdiel medzi najvyššou a najnižšou jasovou hodnotou
- Matematicky:
  ```
  C = (Imax - Imin)/(Imax + Imin)
  ```

b) Úprava kontrastu:
- Zvýšenie:
  ```python
  def increase_contrast(image, factor):
      mean = np.mean(image)
      return (image - mean) * factor + mean
  ```

- Zníženie:
  ```python
  def decrease_contrast(image, factor):
      mean = np.mean(image)
      return (image - mean) / factor + mean
  ```

c) Gamma korekcia:
- Nelineárna transformácia:
  ```
  g = c * f^γ
  kde:
  γ < 1: zvýšenie jasu v tmavých oblastiach
  γ > 1: zvýšenie kontrastu v svetlých oblastiach
  ```

Praktický príklad implementácie gamma korekcie:
```python
def gamma_correction(image, gamma):
    # Normalizácia do rozsahu 0-1
    normalized = image / 255.0
    
    # Aplikácia gamma korekcie
    corrected = np.power(normalized, gamma)
    
    # Späť do rozsahu 0-255
    return (corrected * 255).astype(np.uint8)
```

Dôležité body na zapamätanie:
1. Predspracovanie je kľúčové pre úspešnú analýzu obrazu
2. Rôzne typy šumu vyžadujú rôzne metódy odstránenia
3. Histogram poskytuje dôležité informácie o rozložení jasu v obraze
4. Vyrovnávanie histogramu zlepšuje kontrast obrazu
5. Gamma korekcia je užitočná pre úpravu tmavých alebo svetlých častí obrazu

4. OTAZKA : Geometrické transformácie obrazu, kedy ich možno použiť a ako? Rovnica afinnej transformácie, zápis pre posunutie, zmenu škály. Rotácia. Na čo sa pri geometrických transformáciách používa interpolácia a ako. Základné typy interpolácie - najbližší sused, bilineárna, bikubická. Demonštrujte na príklade. Akým spôsobom možno počítať vzdialenosť, uveďte výpočet Euklidovskej vzdialenosti, city block a chessboard vzdialenosti, na čo sa pri interpolácii vzdialenosť počíta a medzi čím?

Pripravím komplexné vysvetlenie geometrických transformácií a súvisiacich konceptov.

1. Geometrické transformácie obrazu

Využitie:
- Korekcia skreslenia
- Zmena veľkosti obrazu
- Rotácia obrazu
- Registrácia obrazov
- Mapovanie textúr

2. Afinná transformácia

Základná rovnica:
```
[x']   [a11 a12 tx] [x]
[y'] = [a21 a22 ty] [y]
[1 ]   [0   0   1 ] [1]
```

a) Posunutie (translácia):
```
[x']   [1 0 tx] [x]
[y'] = [0 1 ty] [y]
[1 ]   [0 0 1 ] [1]
```

b) Zmena mierky:
```
[x']   [sx 0  0] [x]
[y'] = [0  sy 0] [y]
[1 ]   [0  0  1] [1]
```

c) Rotácia o uhol θ:
```
[x']   [cos(θ) -sin(θ) 0] [x]
[y'] = [sin(θ)  cos(θ) 0] [y]
[1 ]   [0       0      1] [1]
```

3. Interpolácia

Používa sa pri transformáciách na výpočet hodnôt pixelov v novej pozícii.

a) Metóda najbližšieho suseda:
```python
def nearest_neighbor(image, x, y):
    return image[round(y), round(x)]
```

b) Bilineárna interpolácia:
```python
def bilinear_interpolation(image, x, y):
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    
    # Váhy
    wx = x - x1
    wy = y - y1
    
    # Interpolácia
    value = (1-wx)*(1-wy)*image[y1,x1] + \
            wx*(1-wy)*image[y1,x2] + \
            (1-wx)*wy*image[y2,x1] + \
            wx*wy*image[y2,x2]
    
    return value
```

c) Bikubická interpolácia:
- Používa 16 okolotých bodov (4x4)
- Poskytuje hladšie výsledky ale je výpočtovo náročnejšia

4. Výpočet vzdialeností

a) Euklidovská vzdialenosť:
```python
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
```

b) City block (Manhattan) vzdialenosť:
```python
def city_block_distance(x1, y1, x2, y2):
    return abs(x2-x1) + abs(y2-y1)
```

c) Chessboard vzdialenosť:
```python
def chessboard_distance(x1, y1, x2, y2):
    return max(abs(x2-x1), abs(y2-y1))
```

Praktický príklad rotácie obrazu:
```python
def rotate_image(image, angle_degrees):
    # Konverzia uhla na radiány
    angle_radians = np.radians(angle_degrees)
    
    # Výpočet stredu obrazu
    height, width = image.shape[:2]
    center_x, center_y = width/2, height/2
    
    # Rotačná matica
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])
    
    # Nový obraz
    result = np.zeros_like(image)
    
    # Aplikácia rotácie s bilineárnou interpoláciou
    for y in range(height):
        for x in range(width):
            # Posun do stredu
            pos = np.array([x - center_x, y - center_y])
            # Rotácia
            new_pos = rotation_matrix @ pos
            # Posun späť
            new_x = new_pos[0] + center_x
            new_y = new_pos[1] + center_y
            
            # Kontrola hraníc
            if 0 <= new_x < width-1 and 0 <= new_y < height-1:
                result[y,x] = bilinear_interpolation(image, new_x, new_y)
    
    return result
```

Príklad použitia rôznych typov vzdialeností pri interpolácii:
```python
def weighted_interpolation(image, x, y, distance_type='euclidean'):
    # Nájdenie 4 najbližších bodov
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    
    points = [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
    weights = []
    
    # Výpočet váh podľa zvolenej metriky
    for px, py in points:
        if distance_type == 'euclidean':
            d = euclidean_distance(x, y, px, py)
        elif distance_type == 'city_block':
            d = city_block_distance(x, y, px, py)
        else:  # chessboard
            d = chessboard_distance(x, y, px, py)
        
        # Prevod vzdialenosti na váhu
        weights.append(1 / (d + 0.0001))  # Prevencia delenia nulou
    
    # Normalizácia váh
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Výpočet interpolovanej hodnoty
    value = 0
    for (px,py), w in zip(points, weights):
        value += image[py,px] * w
        
    return value
```

Kľúčové body na zapamätanie:
1. Afinné transformácie zachovávajú rovnobežnosť línií
2. Výber interpolačnej metódy ovplyvňuje kvalitu výsledného obrazu
3. Rôzne metriky vzdialenosti môžu byť vhodné pre rôzne aplikácie
4. Bilineárna interpolácia poskytuje dobrý kompromis medzi kvalitou a výpočtovou náročnosťou
5. Pri transformáciách je dôležité správne ošetriť hranice obrazu

5. OTAZKA: Gradient v šedotónovom obraze, čo to je a čomu zodpovedá? Ako aproximujeme výpočet gradientu v digitálnom obraze? Čo je to hrana, ako je definovaná, uveďte niektoré typy hrán. Čo je to detekcia hrán a ako funguje Cannyho detektor hrán

Pripravím podrobné vysvetlenie gradientu obrazu a detekcie hrán.

1. Gradient v šedotónovom obraze

- Definícia: Gradient predstavuje smer a veľkosť najväčšej zmeny jasu v obraze
- Matematický zápis:
```
∇f = [∂f/∂x, ∂f/∂y]
kde:
∂f/∂x - parciálna derivácia v smere x
∂f/∂y - parciálna derivácia v smere y
```

- Veľkosť (magnitúda) gradientu:
```
|∇f| = √[(∂f/∂x)² + (∂f/∂y)²]
```

- Smer gradientu:
```
θ = arctan(∂f/∂y / ∂f/∂x)
```

2. Aproximácia gradientu v digitálnom obraze

a) Základné operátory:
- Robertsov operátor:
```
Gx = [+1  0] Gy = [ 0 +1]
     [ 0 -1]      [-1  0]
```

- Sobelov operátor:
```
Gx = [-1 0 +1] Gy = [+1 +2 +1]
     [-2 0 +2]      [ 0  0  0]
     [-1 0 +1]      [-1 -2 -1]
```

- Prewittovej operátor:
```
Gx = [-1 0 +1] Gy = [+1 +1 +1]
     [-1 0 +1]      [ 0  0  0]
     [-1 0 +1]      [-1 -1 -1]
```

3. Hrany v obraze

a) Definícia:
- Hrana je miesto v obraze, kde dochádza k výraznej zmene jasovej funkcie

b) Typy hrán:
1. Skoková hrana (step edge):
```
Jasová funkcia:  _____
                     |_____
```

2. Líniová hrana (line edge):
```
Jasová funkcia:  _____
                     |‾‾‾‾‾
```

3. Strešná hrana (roof edge):
```
Jasová funkcia:      /\
                 ___/  \___
```

4. Rampová hrana (ramp edge):
```
Jasová funkcia:  ____
                    /____
```

4. Cannyho detektor hrán

Postup:
1. Redukcia šumu (Gaussov filter)
```python
def gaussian_filter(image, sigma):
    size = int(6 * sigma + 1)
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i,j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)
```

2. Výpočet gradientu (Sobelove operátory):
```python
def compute_gradient(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction
```

3. Potlačenie nemaximálnych hodnôt (Non-maximum suppression):
```python
def non_max_suppression(magnitude, direction):
    height, width = magnitude.shape
    result = np.zeros_like(magnitude)
    
    # Kvantizácia smeru do 4 smerov (0°, 45°, 90°, 135°)
    direction = np.rad2deg(direction) % 180
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Porovnanie s dvoma susednými pixelmi v smere gradientu
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                neighbors = [magnitude[i,j+1], magnitude[i,j-1]]
            elif (22.5 <= direction[i,j] < 67.5):
                neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
            elif (67.5 <= direction[i,j] < 112.5):
                neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
            else:
                neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
            
            if magnitude[i,j] >= max(neighbors):
                result[i,j] = magnitude[i,j]
    
    return result
```

4. Dvojité prahovanie (Double thresholding):
```python
def double_thresholding(image, low_ratio=0.05, high_ratio=0.15):
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    
    return strong_edges, weak_edges
```

5. Sledovanie hrán hysteréziou (Edge tracking by hysteresis):
```python
def edge_tracking(strong_edges, weak_edges):
    height, width = strong_edges.shape
    result = np.copy(strong_edges)
    
    # Sledovanie slabých hrán pripojených k silným
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak_edges[i,j]:
                # Kontrola 8-okolia
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    result[i,j] = True
    
    return result
```

Kompletný Cannyho detektor:
```python
def canny_detector(image, sigma=1, low_ratio=0.05, high_ratio=0.15):
    # 1. Redukcia šumu
    smoothed = gaussian_filter(image, sigma)
    
    # 2. Výpočet gradientu
    magnitude, direction = compute_gradient(smoothed)
    
    # 3. Potlačenie nemaximálnych hodnôt
    suppressed = non_max_suppression(magnitude, direction)
    
    # 4. Dvojité prahovanie
    strong_edges, weak_edges = double_thresholding(suppressed, 
                                                 low_ratio, 
                                                 high_ratio)
    
    # 5. Sledovanie hrán
    edges = edge_tracking(strong_edges, weak_edges)
    
    return edges
```

Kľúčové body na zapamätanie:
1. Gradient poskytuje informácie o smere a veľkosti zmeny jasu
2. Existujú rôzne typy hrán v závislosti od priebehu jasovej funkcie
3. Cannyho detektor je robustný algoritmus zahŕňajúci viacero krokov
4. Výber prahov v Cannyho detektore významne ovplyvňuje výsledok
5. Potlačenie nemaximálnych hodnôt je kľúčové pre získanie tenkých hrán

6. OTAZKA: Ako je definovaná operácia konvolúcie pre digitálne obrazy? Popíšte aplikáciu vybranej konvolučnej masky na obraz. Aké sú masky pre vyhladzovanie a pre ostrenie obrazu (zvýraznenie hrán, všesmerové a smerové)? Ako možno konvolúciu využiť na vyhľadávanie vzoru? Kedy a prečo to funguje, aké sú obmedzenia?

Pripravím komplexné vysvetlenie konvolúcie a jej využitia v spracovaní obrazu.

1. Definícia konvolúcie pre digitálne obrazy

Matematický zápis:
```
g(x,y) = f(x,y) * h(x,y) = Σ Σ f(i,j) * h(x-i, y-j)
```
kde:
- f(x,y) je vstupný obraz
- h(x,y) je konvolučná maska (kernel)
- g(x,y) je výstupný obraz

Implementácia konvolúcie:
```python
def convolution2d(image, kernel):
    # Rozmery obrazu a kernelu
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Padding
    pad_h = k_height // 2
    pad_w = k_width // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'edge')
    
    # Výstupný obraz
    output = np.zeros_like(image)
    
    # Konvolúcia
    for y in range(i_height):
        for x in range(i_width):
            # Výrez obrazu pod kernelom
            region = padded[y:y+k_height, x:x+k_width]
            # Aplikácia kernelu
            output[y,x] = np.sum(region * kernel)
            
    return output
```

2. Konvolučné masky

a) Vyhladzovanie obrazu:
- Priemerovacia maska (box filter):
```
[1/9 1/9 1/9]
[1/9 1/9 1/9]
[1/9 1/9 1/9]
```

- Gaussova maska (σ=1):
```
[1/16 2/16 1/16]
[2/16 4/16 2/16]
[1/16 2/16 1/16]
```

b) Ostrenie obrazu (zvýraznenie hrán):
- Laplacián (všesmerový):
```
[0  1  0]
[1 -4  1]
[0  1  0]
```

- Sobelove masky (smerové):
Horizontálna:
```
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```
Vertikálna:
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

3. Vyhľadávanie vzoru pomocou konvolúcie

Princíp:
```python
def template_matching(image, template):
    # Normalizácia template
    template = template - np.mean(template)
    template = template / np.std(template)
    
    # Konvolúcia
    result = convolution2d(image, template)
    
    # Normalizácia výsledku
    result = (result - np.mean(result)) / np.std(result)
    
    return result
```

Príklad použitia:
```python
def find_pattern(image, pattern, threshold=0.8):
    # Aplikácia template matching
    correlation = template_matching(image, pattern)
    
    # Nájdenie maxím
    locations = np.where(correlation > threshold)
    
    return list(zip(locations[1], locations[0]))  # x, y koordináty
```

Obmedzenia template matchingu:
1. Citlivosť na rotáciu
```python
# Riešenie - vytvorenie rotovaných verzií vzoru
def create_rotated_templates(template, angles):
    templates = []
    for angle in angles:
        rotated = rotate_image(template, angle)
        templates.append(rotated)
    return templates
```

2. Citlivosť na zmenu mierky
```python
# Riešenie - vytvorenie vzoru v rôznych mierkach
def create_scaled_templates(template, scales):
    templates = []
    for scale in scales:
        scaled = cv2.resize(template, None, fx=scale, fy=scale)
        templates.append(scaled)
    return templates
```

3. Citlivosť na zmenu osvetlenia
```python
# Riešenie - normalizácia jasu
def normalize_brightness(image):
    return (image - np.mean(image)) / np.std(image)
```

4. Praktické použitie konvolúcie

a) Implementácia rôznych filtrov:
```python
def apply_filter(image, filter_type):
    if filter_type == 'blur':
        kernel = np.ones((3,3)) / 9
    elif filter_type == 'sharpen':
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
    elif filter_type == 'edge_detect':
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        
    return convolution2d(image, kernel)
```

b) Kombinácia filtrov:
```python
def enhance_image(image):
    # Najprv redukcia šumu
    blurred = apply_filter(image, 'blur')
    
    # Potom zvýraznenie hrán
    enhanced = apply_filter(blurred, 'sharpen')
    
    return enhanced
```

Kľúčové body na zapamätanie:
1. Konvolúcia je základná operácia v spracovaní obrazu
2. Rôzne masky poskytujú rôzne efekty (vyhladzovanie, ostrenie)
3. Template matching je citlivý na rotáciu, mierku a osvetlenie
4. Pre robustné vyhľadávanie vzoru je potrebné použiť viacero variant vzoru
5. Správny výber konvolučnej masky závisí od konkrétnej aplikácie

7. OTAZKA: Definujte Fourierovu transformáciu, čo je jej vstupom a čo výstupom? Aké sú metódy zobrazenia frekvenčného spektra? Čomu zodpovedajú frekvencie ak hovoríme o 2D obraze? Akým spôsobom môžeme použiť filtrovanie vo frekvenčnom spektre a aký je celý postup pre získanie filtrovaného obrazu? Aký je vzťah konvolúcie a Fourierovej transformácie? Na aké účely možno ešte použiť analýzu frekvenčného spektra

Pripravím komplexné vysvetlenie Fourierovej transformácie a jej využitia v spracovaní obrazu.

1. Fourierova transformácia (FT)

Definícia pre 2D obraz:
```
F(u,v) = ∫∫ f(x,y) * e^(-j2π(ux+vy)) dx dy
```
kde:
- f(x,y) je vstupný obraz v priestorovej doméne
- F(u,v) je obraz vo frekvenčnej doméne
- u,v sú frekvenčné premenné

Diskrétna Fourierova transformácia (DFT):
```python
def compute_dft(image):
    # Použitie FFT algoritmu
    f_transform = np.fft.fft2(image)
    # Presun nulových frekvencií do stredu
    f_transform_shifted = np.fft.fftshift(f_transform)
    return f_transform_shifted
```

2. Zobrazenie frekvenčného spektra

a) Amplitúdové spektrum:
```python
def display_spectrum(f_transform):
    magnitude_spectrum = np.abs(f_transform)
    # Logaritmická transformácia pre lepšiu vizualizáciu
    log_spectrum = np.log1p(magnitude_spectrum)
    return log_spectrum

def normalize_spectrum(spectrum):
    return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
```

b) Fázové spektrum:
```python
def display_phase(f_transform):
    phase_spectrum = np.angle(f_transform)
    return phase_spectrum
```

3. Význam frekvencií v 2D obraze:

- Nízke frekvencie (stred spektra):
  - Zodpovedajú pomaly sa meniacim jasom
  - Reprezentujú hrubé štruktúry a pozadie

- Vysoké frekvencie (okraje spektra):
  - Zodpovedajú rýchlym zmenám jasu
  - Reprezentujú hrany a detaily

4. Filtrovanie vo frekvenčnej doméne

a) Dolnopriepustný filter (potlačenie vysokých frekvencií):
```python
def lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    
    # Vytvorenie masky
    mask = np.zeros((rows, cols))
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x*x + y*y <= cutoff*cutoff
    mask[mask_area] = 1
    
    return mask
```

b) Hornopriepustný filter (potlačenie nízkych frekvencií):
```python
def highpass_filter(shape, cutoff):
    return 1 - lowpass_filter(shape, cutoff)
```

Celý proces filtrovania:
```python
def frequency_domain_filter(image, filter_mask):
    # 1. Výpočet DFT
    f_transform = compute_dft(image)
    
    # 2. Aplikácia filtra
    filtered_f = f_transform * filter_mask
    
    # 3. Inverzná DFT
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))
    
    return filtered_image
```

5. Vzťah konvolúcie a Fourierovej transformácie

Konvolučný teorém:
```
f(x,y) * h(x,y) ←→ F(u,v) · H(u,v)
```
kde:
- * označuje konvolúciu
- · označuje násobenie po prvkoch
- F(u,v) je FT obrazu
- H(u,v) je FT konvolučnej masky

Implementácia:
```python
def convolution_frequency(image, kernel):
    # FT obrazu
    F = np.fft.fft2(image)
    # FT kernelu (potrebné doplniť na veľkosť obrazu)
    H = np.fft.fft2(kernel, s=image.shape)
    # Násobenie vo frekvenčnej doméne
    G = F * H
    # Inverzná FT
    result = np.real(np.fft.ifft2(G))
    return result
```

6. Ďalšie využitie frekvenčného spektra

a) Analýza textúr:
```python
def analyze_texture(image):
    f_transform = compute_dft(image)
    spectrum = np.abs(f_transform)
    
    # Rozdelenie spektra na prstence
    rings = []
    center_y, center_x = spectrum.shape[0]//2, spectrum.shape[1]//2
    
    for r in range(0, min(center_x, center_y), 10):
        y, x = np.ogrid[-center_y:spectrum.shape[0]-center_y,
                       -center_x:spectrum.shape[1]-center_x]
        mask = (x*x + y*y >= r**2) & (x*x + y*y < (r+10)**2)
        rings.append(np.mean(spectrum[mask]))
        
    return rings  # Charakteristika textúry
```

b) Detekcia periodických vzorov:
```python
def detect_patterns(image):
    f_transform = compute_dft(image)
    spectrum = np.abs(f_transform)
    
    # Hľadanie lokálnych maxím v spektre
    from scipy.signal import find_peaks_cwt
    peaks = find_peaks_cwt(spectrum.ravel(), np.arange(1,10))
    
    return peaks
```

Kľúčové body na zapamätanie:
1. FT rozkladá obraz na frekvenčné komponenty
2. Nízke frekvencie zodpovedajú hrubým štruktúram, vysoké detailom
3. Filtrovanie vo frekvenčnej doméne je ekvivalentné konvolúcii
4. Frekvenčné spektrum poskytuje užitočné informácie o štruktúre obrazu
5. FT je reverzibilná - môžeme sa vrátiť do priestorovej domény

8. OTAZKA: Čo je to segmentácia obrazu? Segmentácia obrazu podľa jasu, prahovanie jedným, dvomi, viacerými prahmi. Metódy automatického určovania prahu - iteratívny algoritmus hľadania optimálneho prahu. Čo je to adaptívne prahovanie, uveďte príklad aplikácie adaptívneho prahovania na segmentáciu textu z nerovnomerne osvetleného povrchu

Pripravím komplexné vysvetlenie segmentácie obrazu a rôznych metód prahovania.

1. Segmentácia obrazu
- Cieľ: rozdelenie obrazu na oblasti so spoločnými vlastnosťami
- Výstup: obraz, kde každý pixel patrí do určitej triedy/segmentu
- Využitie: rozpoznávanie objektov, analýza scény, spracovanie dokumentov

2. Segmentácia podľa jasu (Prahovanie)

a) Prahovanie jedným prahom:
```python
def simple_threshold(image, threshold):
    return np.where(image > threshold, 255, 0)
```

b) Prahovanie dvoma prahmi:
```python
def dual_threshold(image, low_threshold, high_threshold):
    result = np.zeros_like(image)
    result[(image >= low_threshold) & (image <= high_threshold)] = 255
    return result
```

c) Prahovanie viacerými prahmi:
```python
def multi_threshold(image, thresholds):
    result = np.zeros_like(image)
    for i, threshold in enumerate(thresholds[:-1]):
        mask = (image >= threshold) & (image < thresholds[i+1])
        result[mask] = i * (255 // (len(thresholds)-1))
    return result
```

3. Automatické určovanie prahu

Iteratívny algoritmus (Otsu):
```python
def otsu_threshold(image):
    # Výpočet histogramu
    hist = np.histogram(image, bins=256, range=[0,256])[0]
    hist = hist / hist.sum()
    
    # Inicializácia
    best_threshold = 0
    best_variance = 0
    
    # Prechod všetkými možnými prahmi
    for threshold in range(1, 255):
        # Rozdelenie histogramu
        w0 = hist[:threshold].sum()
        w1 = hist[threshold:].sum()
        
        if w0 == 0 or w1 == 0:
            continue
            
        # Výpočet priemerov
        mu0 = np.average(range(threshold), weights=hist[:threshold])
        mu1 = np.average(range(threshold, 256), weights=hist[threshold:])
        
        # Výpočet medzi-triednej variancie
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        # Aktualizácia najlepšieho prahu
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
            
    return best_threshold
```

4. Adaptívne prahovanie

Implementácia adaptívneho prahovania:
```python
def adaptive_threshold(image, window_size=11, C=2):
    # Vytvorenie prázdneho výstupného obrazu
    result = np.zeros_like(image)
    
    # Padding obrazu
    pad = window_size // 2
    padded = np.pad(image, pad, mode='edge')
    
    # Prechod obrazom
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Výrez okna
            window = padded[i:i+window_size, j:j+window_size]
            # Výpočet lokálneho prahu
            local_threshold = np.mean(window) - C
            # Prahovanie
            result[i,j] = 255 if image[i,j] > local_threshold else 0
            
    return result
```

Vylepšená verzia pre nerovnomerne osvetlený text:
```python
def adaptive_document_threshold(image, window_size=51, C=10):
    # Gaussovo rozmazanie pre odhad pozadia
    blur = cv2.GaussianBlur(image, (window_size, window_size), 0)
    
    # Adaptívne prahovanie pomocou lokálneho kontrastu
    threshold = blur - C
    result = np.where(image > threshold, 255, 0)
    
    return result.astype(np.uint8)
```

Príklad použitia pre dokument:
```python
def process_document(image_path):
    # Načítanie obrazu
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplikácia adaptívneho prahovania
    binary = adaptive_document_threshold(image)
    
    # Odstránenie šumu
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
```

5. Praktické príklady využitia:

a) Segmentácia textu z dokumentu s tieňom:
```python
def segment_shadowed_document(image):
    # Normalizácia kontrastu
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Adaptívne prahovanie s veľkým oknom
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=25
    )
    
    return binary
```

b) Segmentácia s premenlivým osvetlením:
```python
def segment_variable_lighting(image):
    # Odhad pozadia
    background = cv2.medianBlur(image, 51)
    
    # Korekcia osvetlenia
    corrected = cv2.subtract(image, background)
    normalized = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    
    # Prahovanie
    return adaptive_threshold(normalized)
```

Kľúčové body na zapamätanie:
1. Globálne prahovanie funguje dobre len pri rovnomernom osvetlení
2. Otsu metóda je efektívna pre bimodálne histogramy
3. Adaptívne prahovanie je vhodné pre nerovnomerne osvetlené obrazy
4. Veľkosť okna pri adaptívnom prahovaní ovplyvňuje výsledok
5. Pre dokumenty je často potrebné kombinovať viacero techník

9. OTAZKA: 
