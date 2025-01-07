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

9. OTAZKA: Farba a farebné modely (aditívne a subtraktívne), porovnajte RGB a Lab model z hľadiska segmentácie objektov podľa farby, vysvetlite segmentáciu podľa farby s použitím Lab farebného modelu, možnosti vymedzenia oblastí zodpovedajúcich konkrétnej farbe. Segmentácia štiepením a spájaním oblastí. Ako postupujeme pri jednotlivých metódach, ako je ich možné kombinovať. Čo je to “kritérium homogenity”, uveďte príklad. Ako funguje pyramid linking?

### **Farba a farebné modely (aditívne a subtraktívne)**

**Aditívne farebné modely (RGB):**
- Používajú sa tam, kde je zdrojom farby svetlo, napr. v monitoroch, televíziách, projektoroch.
- Základné farby: **červená (Red), zelená (Green), modrá (Blue)**.
- Pri ich kombinácii vznikajú ostatné farby: 
  - Červená + zelená = žltá
  - Červená + modrá = purpurová
  - Zelená + modrá = tyrkysová
  - Všetky tri dohromady = biela.
- Model je vhodný na zobrazenie farieb na zariadeniach, ktoré vyžadujú emisiu svetla.

**Subtraktívne farebné modely (CMY/CMYK):**
- Používajú sa pri tlači a v prípade pigmentov, kde farba vzniká absorbovaním (subtrakciou) svetla.
- Základné farby: **azúrová (Cyan), purpurová (Magenta), žltá (Yellow)**. 
- Pri tlači sa pridáva čierna (Key/Black – CMYK), aby bola čierna farba presnejšia.
- Kombinácia všetkých troch pigmentov teoreticky vytvára čiernu, no v praxi je to tmavohnedá.

---

### **Porovnanie RGB a Lab modelu z hľadiska segmentácie objektov podľa farby**

1. **RGB farebný model:**
   - Je vhodný na zobrazovanie farieb, ale menej vhodný na segmentáciu, pretože je závislý od osvetlenia.
   - Farby v RGB nie sú lineárne príbuzné vnímaniu farieb človekom – malé zmeny hodnôt môžu mať nepredvídateľný vizuálny efekt.
   - Napríklad tieň alebo zmena intenzity osvetlenia ovplyvňuje hodnoty R, G a B, čo komplikuje segmentáciu.

2. **Lab farebný model:**
   - Lab model je percepčne založený a nezávisí od osvetlenia (pokiaľ je normalizovaný).
   - Obsahuje:
     - **L** (Lightness) – jasová zložka.
     - **a** (Green to Magenta) – farebný kanál.
     - **b** (Blue to Yellow) – farebný kanál.
   - Lepšie oddeľuje farebné informácie od intenzity osvetlenia, čo uľahčuje segmentáciu objektov na základe farby, najmä v nehomogénne osvetlených scénach.

---

### **Segmentácia podľa farby s použitím Lab farebného modelu**

1. **Postup:**
   - Prevedieme obrázok z RGB modelu do Lab modelu.
   - Použijeme len farebné kanály **a** a **b**, čím odstránime vplyv osvetlenia.
   - Definujeme hranice pre farebné hodnoty **a** a **b**, ktoré odpovedajú farbe záujmu (napr. červená farba má hodnoty a > 20, b < 20).
   - Na základe týchto hraníc identifikujeme pixely zodpovedajúce farbe záujmu.

2. **Príklad:**
   - Ak chceme nájsť červené oblasti:
     - Pre každý pixel vypočítame hodnoty **a** a **b**.
     - Ak hodnoty spadajú do rozsahu charakteristického pre červenú farbu, pixel označíme ako patriaci k červenej oblasti.

3. **Vymedzenie oblastí podľa farby:**
   - Používajú sa prahovacie techniky (thresholding).
   - Napr. pixely s hodnotami **a > X1** a **b < X2** označíme ako „červené“.
   - Výstupom je binárna maska, kde 1 reprezentuje oblasť záujmu.

---

### **Segmentácia štiepením a spájaním oblastí**

1. **Štiepenie (Splitting):**
   - Rozdeľujeme obrázok na menšie homogénne časti.
   - Používa sa, ak veľké oblasti nie sú homogénne.
   - Postup:
     - Začneme s celou oblasťou.
     - Skontrolujeme, či spĺňa kritérium homogenity.
     - Ak nie, oblasť rozdelíme (napr. na štvrtiny).
     - Opakujeme, kým všetky oblasti nie sú homogénne.

2. **Spájanie (Merging):**
   - Spojujeme malé oblasti do väčších, ak spĺňajú kritérium homogenity.
   - Používa sa po štiepení na optimalizáciu segmentácie.
   - Postup:
     - Skontrolujeme susedné oblasti.
     - Ak majú podobné hodnoty (napr. farebné alebo textúrne), spojíme ich do jednej oblasti.

3. **Kombinácia metód:**
   - **Štiepenie** použijeme na vytvorenie základných homogénnych oblastí.
   - **Spájanie** následne spojí susedné homogénne oblasti, čím minimalizuje prebytočné segmenty.

---

### **Kritérium homogenity**

- **Definícia:** Podmienka, ktorá určuje, či je oblasť dostatočne jednotná na to, aby bola považovaná za homogénnu.
- Môže byť založené na:
  - Hodnotách farby (napr. variácia vo farebných kanáloch < prah).
  - Textúre (napr. rovnaká frekvencia pixelov v oblasti).
  - Jasnosti (napr. priemerná intenzita v oblasti ± odchýlka).
- **Príklad:** Oblasť je homogénna, ak variácia hodnoty „a“ a „b“ v Lab modeli je menšia ako 5.

---

### **Pyramid Linking**

- **Definícia:** Metóda, ktorá spája informácie z viacerých úrovní rozlíšenia obrázka.
- Postup:
  - Vytvorí sa pyramída obrázkov:
    - Najvyššia úroveň má najnižšie rozlíšenie (downsampling).
    - Najnižšia úroveň má pôvodné rozlíšenie.
  - Na každej úrovni sa vykoná segmentácia.
  - Výsledky z vyšších úrovní sa použijú na usmernenie segmentácie na nižších úrovniach.
- **Príklad:** 
  - Pri vyhľadávaní veľkých objektov sa použijú vyššie úrovne (nízke rozlíšenie).
  - Pri detailnom vyhľadávaní menších častí sa použijú nižšie úrovne (vysoké rozlíšenie).

---

Tento prehľad obsahuje teoretické základy aj praktické príklady, ktoré ti pomôžu pri skúške. Ak potrebuješ prepočítať konkrétny algoritmus alebo príklady kódu, stačí dať vedieť. 😊

10. OTAZKA: Označovanie spojených komponentov, uveďte príklad algoritmu. Morfologické operácie, dilatácia a erózia (štruktúrny element, kritériá hit a fit), otvorenie a uzavretie, transformácia hit-or-miss, vzdialenostná transformácia, skeletonizácia, princíp segmentácie pomocou rozvodia, šedotónová erózia a dilatácia, oddelenie pozadia pomocou šedotónovej erózie

### **Označovanie spojených komponentov**

Označovanie spojených komponentov (Connected Component Labeling, CCL) je metóda na identifikáciu a označovanie súvislých oblastí v binárnom obraze. Používa sa v analýze obrazov na segmentáciu a identifikáciu objektov.

1. **Algoritmus na označovanie spojených komponentov (2-pass algoritmus):**
   - Predpokladáme binárny obraz, kde 1 označuje pixely patriace k objektu a 0 pozadie.
   - Používa 4-susednosť alebo 8-susednosť.

   **Kroky:**
   1. **Priechod 1:**
      - Prechádzame obraz pixel po pixeli.
      - Ak je pixel súčasťou objektu (hodnota 1):
        - Skontrolujeme jeho susedov (v závislosti od susednosti, napr. 4-susednosť kontroluje horného a ľavého suseda).
        - Ak majú susedia už priradené značky, pixel dostane najmenšiu z týchto značiek.
        - Ak nemá žiadny sused značku, pixel dostane novú značku.
        - Zaznamenávame ekvivalencie medzi značkami (ak rôzni susedia majú rôzne značky).
   2. **Priechod 2:**
      - Všetky pixely s ekvivalentnými značkami sa aktualizujú na rovnakú konečnú značku.
   3. **Výstup:**
      - Každá súvislá oblasť v obraze má jedinečnú značku.

   **Príklad:**
   Binárny obraz:
   ```
   0 1 1 0
   1 1 0 0
   0 0 1 1
   0 1 1 0
   ```
   Po aplikovaní algoritmu (4-susednosť):
   ```
   0 1 1 0
   1 1 0 0
   0 0 2 2
   0 2 2 0
   ```

---

### **Morfologické operácie**

Morfologické operácie sa používajú na spracovanie binárnych alebo šedotónových obrazov. Využívajú **štruktúrny element (SE)**, ktorý definuje spôsob interakcie s obrazom.

#### **1. Dilatácia:**
- Rozširuje objekty v obraze, pridáva pixely k hranám objektu.
- **Hit kritérium:** Štruktúrny element sa musí aspoň čiastočne prekrývať s objektom.
- **Vzorec (binárne obrazy):**  
  \( A \oplus B = \{ z \mid (B_z \cap A) \neq \emptyset \} \)

#### **2. Erózia:**
- Zmenšuje objekty v obraze, odstraňuje pixely na hranách objektu.
- **Fit kritérium:** Štruktúrny element sa musí úplne vojsť do objektu.
- **Vzorec (binárne obrazy):**  
  \( A \ominus B = \{ z \mid B_z \subseteq A \} \)

#### **3. Otvorenie (Opening):**
- Sekvencia erózie a dilatácie.
- Používa sa na odstránenie šumu a malých objektov.
- \( A \circ B = (A \ominus B) \oplus B \)

#### **4. Uzavretie (Closing):**
- Sekvencia dilatácie a erózie.
- Používa sa na vyplnenie medzier a spájanie objektov.
- \( A \bullet B = (A \oplus B) \ominus B \)

---

### **Transformácia hit-or-miss**
- Identifikuje konkrétne tvarové vzory v obraze.
- Využíva **štruktúrny element** \( B \) pozostávajúci z dvoch častí:
  - Oblasť, ktorá musí obsahovať objekt.
  - Oblasť, ktorá musí byť pozadím.
- **Vzorec:**  
  \( A \otimes B = (A \ominus B_1) \cap (A^c \ominus B_2) \), kde \( B_1 \) je objektová a \( B_2 \) pozadová časť SE.

---

### **Vzdialenostná transformácia**
- Pre každý pixel v obraze vypočíta vzdialenosť k najbližšiemu pixelu pozadia.
- Používa sa na:
  - Návrh štruktúr.
  - Skeletonizáciu.
  - Segmentáciu pomocou rozvodia.
- Príklad metriky: **Euklidovská vzdialenosť**, **Manhattanská vzdialenosť**.

---

### **Skeletonizácia**
- Zredukuje objekty na ich kostru (jednobunkovú hrúbku), pričom zachová ich topológiu.
- Použitie: Rekonštrukcia tvarov, analýza tvarových vlastností.

---

### **Princíp segmentácie pomocou rozvodia**
- Modeluje obraz ako topografickú mapu:
  - Jas predstavuje výšku.
- Simuluje zaplavenie:
  - Voda tečie z minima smerom k maximám.
  - Oblasti, ktoré sa zaplavujú z rôznych minim, sa oddelia „hradnými múrmi“.
- Používa sa na segmentáciu objektov na základe hraničných hodnôt.

---

### **Šedotónová erózia a dilatácia**
1. **Šedotónová dilatácia:**
   - Nahradí pixel maximálnou hodnotou v jeho okolí definovanom štruktúrnym elementom.
2. **Šedotónová erózia:**
   - Nahradí pixel minimálnou hodnotou v jeho okolí definovanom štruktúrnym elementom.

---

### **Oddelenie pozadia pomocou šedotónovej erózie**
- Použitie: Identifikácia objektov na pozadí.
- Postup:
  - Aplikujeme šedotónovú eróziu na obraz.
  - Pozadie sa „odstráni“, pretože sa znížia hodnoty intenzity.

---

Tento obsah obsahuje kľúčové koncepty, vzorce a príklady potrebné na pochopenie témy a prípravu na skúšku. Ak potrebuješ viac detailov alebo kódové implementácie, daj vedieť! 😊

11 OTAZKA: Rozpoznávanie objektov, jednoduché skalárne deskriptory oblastí, klasifikátory, pravidlový klasifikátor, diskriminatívne a generatívne metódy rozpoznávania, Bayesovo pravidlo a naivný Bayesov klasifikátor

Pripravím pre vás komplexnú odpoveď, ktorá vám pomôže pripraviť sa na skúšku. Rozoberiem každú časť podrobne:

1. Rozpoznávanie objektov:
- Je to proces určovania, či obrazové dáta obsahujú špecifický objekt, vlastnosť alebo aktivitu
- Delí sa na:
  - Objektové rozpoznávanie (klasifikácia) - rozpoznávanie jedného alebo viacerých naučených objektov vrátane ich 2D pozície v obraze alebo 3D pózy
  - Identifikácia - rozpoznávanie konkrétnej inštancie objektu (napr. konkrétna tvár osoby, odtlačok prsta)
  - Detekcia - skenovanie obrazu pre špecifické podmienky (napr. abnormálne bunky v medicínskych snímkach)

2. Jednoduché skalárne deskriptory oblastí:
- Veľkosť/Plocha (Area):
  - Počet pixelov vnútri oblasti
  - Pri polygóne sa dá vypočítať zo súradníc vrcholov

- Obvod (Perimeter):
  - Súčet Euklidovských vzdialeností obrysových bodov pozdĺž kontúry

- Okrúhlosť (Roundness):
  - Vzorec: (4 × π × plocha) / obvod²
  - Hodnoty v intervale <0,1>, kde 1.0 predstavuje dokonalý kruh

- Feretov priemer:
  - Priemer kruhu s rovnakou plochou ako sledovaný objekt

- Dĺžka hlavnej osi:
  - Najdlhšia úsečka medzi dvoma ľubovoľnými bodmi obvodu

- Eulerovo číslo:
  - E = S - N (S = počet súvislých objektov, N = počet dier)
  - Popisuje topológiu objektu

3. Pravidlový klasifikátor:
Príklad jednoduchých pravidiel:
```
Ak (Okrúhlosť > 0.9 AND Plocha > 50) potom Objekt = Jablko
inak Ak (Okrúhlosť > 0.7 AND Plocha > 50) potom Objekt = Hruška
inak Ak (Okrúhlosť < 0.7 AND Plocha < 30) potom Objekt = Slivka
inak Objekt = Neznámy
```

4. Diskriminatívne metódy rozpoznávania:
- Pracujú v priestore príznakov (N-rozmernom)
- Vytvárajú model určujúci hranicu medzi triedami (diskriminačná funkcia)
- Potrebujú príklady aspoň dvoch tried
- Príklady metód:
  - Umelé neurónové siete (ANN)
  - Support Vector Machines (SVM)
  - Rozhodovacie stromy
  - Lineárna regresia

5. Generatívne metódy rozpoznávania:
- Model vymedzuje hranice jednotlivých tried
- Dokážu generovať nové príklady danej triedy
- Stačia príklady jednej triedy
- Príklad: Naivný Bayesov klasifikátor

6. Bayesovo pravidlo:
- P(A|B) = P(B|A) × P(A) / P(B)
- Praktický príklad z prednášky (drogový test):
  - Citlivosť testu: 90%
  - Špecifickosť: 95%
  - Prevalencia: 5%
  - Výpočet pravdepodobnosti užívania drog pri pozitívnom teste:
    P(beriem|pozit. test) = 0.9 × 0.05 / (0.05 × 0.9 + 0.05 × 0.95) = 0.486 (48.6%)

7. Naivný Bayesov klasifikátor:
- Predpoklady:
  - Príznaky sú navzájom nezávislé
  - Hodnoty príznakov majú normálne rozdelenie pravdepodobnosti

Príklad klasifikácie:
- Máme dátovú množinu s príznakmi (napr. okrúhlosť) a triedami (jablko, hruška)
- Pre nový objekt s okrúhlosťou 0.88:
  1. Vypočítame P(jablko) a P(hruška) z trénovacích dát
  2. Aproximujeme rozdelenie príznakov pomocou Gaussových kriviek
  3. Vypočítame P(okrúhlosť = 0.88|jablko) a P(okrúhlosť = 0.88|hruška)
  4. Pomocou Bayesovho pravidla určíme pravdepodobnosti tried
  5. Klasifikujeme podľa vyššej pravdepodobnosti

Dôležité poznámky pre skúšku:
- Vedieť vysvetliť rozdiely medzi diskriminatívnymi a generatívnymi metódami
- Poznať základné skalárne deskriptory a vedieť ich vypočítať
- Rozumieť Bayesovmu pravidlu a vedieť ho aplikovať na praktické príklady
- Chápať princíp naivného Bayesovho klasifikátora a jeho predpoklady
- Vedieť navrhnúť jednoduchý pravidlový klasifikátor pre konkrétny problém

12 OTAZKA : Základný princíp fungovania metódy Violu-Jonesa, Haarova črta, spájanie výstupov viacerých Haarových čŕt, príklad pre detekciu tvárí. Základný popis fungovania SIFT, čo sú to črty - kľúčové body, deskriptor, škálová a rotačná invariancia, náčrt ako funguje klasifikácia objektu.

Pripravím komplexné vysvetlenie týchto dôležitých metód počítačového videnia:

1. METÓDA VIOLA-JONES
Základný princíp:
- Framework navrhnutý v roku 2001 pre detekciu objektov v reálnom čase
- Primárne určený na detekciu tvárí, ale použiteľný aj pre iné objekty
- Pozostáva zo 4 hlavných fáz:

A) Haarove črty (Haar Features):
- Sú to jednoduché obdĺžnikové filtre pozostávajúce z čiernych a bielych oblastí
- Hodnota črty sa počíta ako rozdiel súm pixelov pod bielymi a čiernymi oblasťami
- Príklady Haarových čŕt pre detekciu tvárí:
  - Horizontálna črta pre detekciu očí (tmavšia oblasť medzi svetlejšími)
  - Vertikálna črta pre detekciu nosa
  - Diagonálne črty pre detekciu líc

B) Integrálny obraz:
- Efektívny spôsob výpočtu súm pixelov v obdĺžnikových oblastiach
- Pre každý bod (x,y) obsahuje sumu všetkých pixelov nad a naľavo
- Umožňuje rýchly výpočet Haarových čŕt pomocou 4 prístupov do pamäte

C) AdaBoost tréning:
- Kombinuje mnoho "slabých" klasifikátorov do jedného silného
- Každý slabý klasifikátor používa jednu Haarovu črtu
- Váži klasifikátory podľa ich úspešnosti
- Výsledný silný klasifikátor je váženou sumou slabých klasifikátorov

D) Kaskádové klasifikátory:
- Usporiadanie klasifikátorov do kaskády (postupnosti)
- Začína jednoduchšími klasifikátormi
- Ak región neprejde cez aktuálny stupeň, je okamžite zamietnutý
- Ak prejde, pokračuje na ďalší stupeň
- Zrýchľuje detekciu tým, že rýchlo eliminuje oblasti bez tváre

2. SIFT (Scale Invariant Feature Transform)
Základný princíp:
- Vyvinutý Davidom Lowem v 1999
- Detekuje a popisuje lokálne črty v obrazoch
- Invariantný voči škále, rotácii a čiastočne voči zmene osvetlenia

A) Kľúčové body (Keypoints):
- Významné body v obraze (rohy, hrany, výrazné textúry)
- Detekcia v rôznych škálach pomocou Difference of Gaussian (DoG)
- Výber stabilných bodov, ktoré sa zachovávajú pri transformáciách

B) Deskriptor:
- 128-dimenzionálny vektor popisujúci okolie kľúčového bodu
- Založený na histogramoch gradientov
- Rozdelenie okolia na 4×4 oblasti
- V každej oblasti 8 smerových histogramov

C) Škálová invariancia:
- Dosiahnutá vytvorením scale-space pyramídy
- Detekcia kľúčových bodov vo všetkých škálach
- Automatický výber charakteristickej škály pre každý bod

D) Rotačná invariancia:
- Výpočet dominantnej orientácie pre každý kľúčový bod
- Deskriptor je normalizovaný vzhľadom na túto orientáciu

E) Klasifikácia objektu pomocou SIFT:
1. Detekcia kľúčových bodov v trénovacej aj testovacej snímke
2. Výpočet deskriptorov pre všetky kľúčové body
3. Párovanie zodpovedajúcich bodov medzi snímkami
4. Použitie zovšeobecnenej Houghovej transformácie na potvrdenie zhody
5. Výpočet afinných transformácií objektu

Hlavné výhody SIFT:
- Robustnosť voči zmene mierky a rotácii
- Čiastočná invariancia voči zmene osvetlenia
- Rozpoznávanie čiastočne prekrytých objektov
- Vysoká presnosť párovania čŕt

Možné aplikácie:
- Rozpoznávanie objektov
- Sledovanie objektov vo videu
- Panoramatické spájanie snímok
- 3D rekonštrukcia scény
- Vizuálna lokalizácia robotov

13 OTAZKA: Houghova transformácia, kompletný popis metódy pre vyhľadávanie priamok, popis modifikácie pre detekciu kružníc, praktické použitie
Pripravím komplexné vysvetlenie Houghovej transformácie:

1. ZÁKLADNÝ PRINCÍP HOUGHOVEJ TRANSFORMÁCIE PRE PRIAMKY

A) Transformácia z priestoru obrazu do parametrického priestoru:
- Každý bod (x,y) v obraze môže ležať na nekonečne mnoho priamok
- Priamka je definovaná rovnicou: y = mx + b
- Alternatívna parametrizácia (používanejšia): ρ = x·cos(θ) + y·sin(θ)
  kde: ρ - kolmá vzdialenosť priamky od počiatku
       θ - uhol medzi kolmicou na priamku a osou x

B) Postup detekcie:
1. Vytvorenie akumulátora (2D pole) pre parametre (ρ,θ)
2. Pre každý hranový bod (x,y) v obraze:
   - Pre každý uhol θ vypočítať ρ
   - Inkrementovať hodnotu v akumulátore na pozícii (ρ,θ)
3. Hľadanie lokálnych maxím v akumulátore
   - Maxima reprezentujú najpravdepodobnejšie priamky v obraze

Príklad implementácie pre priamky:
```python
def hough_transform_lines(edge_image):
    height, width = edge_image.shape
    diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.linspace(-diagonal, diagonal, 2*diagonal)
    thetas = np.deg2rad(np.arange(-90, 90))
    
    # Vytvorenie akumulátora
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Súradnice hranových bodov
    y_idxs, x_idxs = np.nonzero(edge_image)
    
    # Hlasovanie
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]
            rho = x*np.cos(theta) + y*np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos-rho))
            accumulator[rho_idx, theta_idx] += 1
            
    return accumulator, rhos, thetas
```

2. MODIFIKÁCIA PRE DETEKCIU KRUŽNÍC

A) Princíp:
- Kružnica je definovaná troma parametrami: (x₀, y₀, r)
  kde: (x₀,y₀) - stred kružnice
       r - polomer
- Parametrická rovnica: (x - x₀)² + (y - y₀)² = r²

B) Postup detekcie:
1. Vytvorenie 3D akumulátora pre parametre (x₀,y₀,r)
2. Pre každý hranový bod (x,y):
   - Pre každý možný polomer r:
     - Vypočítať možné stredy kružnice
     - Inkrementovať hodnoty v akumulátore
3. Nájsť lokálne maximá v 3D akumulátore

Príklad implementácie pre kružnice:
```python
def hough_transform_circles(edge_image, r_min, r_max):
    height, width = edge_image.shape
    accumulator = np.zeros((height, width, r_max - r_min))
    
    # Súradnice hranových bodov
    y_idxs, x_idxs = np.nonzero(edge_image)
    
    # Hlasovanie
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for r in range(r_min, r_max):
            for theta in np.linspace(0, 2*np.pi, 100):
                x0 = x - r*np.cos(theta)
                y0 = y - r*np.sin(theta)
                if 0 <= x0 < width and 0 <= y0 < height:
                    accumulator[int(y0), int(x0), r-r_min] += 1
                    
    return accumulator
```

3. PRAKTICKÉ POUŽITIE

A) Predspracovanie obrazu:
- Konverzia na odtiene šedej
- Detekcia hrán (napr. Cannyho detektor)
- Redukcia šumu (napr. Gaussov filter)

B) Optimalizácie:
- Obmedzenie rozsahu parametrov
- Prahové hodnoty pre akumulátor
- Lokálne maximum s okolím

C) Aplikácie:
- Detekcia čiar na cestách
- Rozpoznávanie dopravných značiek
- Detekcia objektov v priemysle
- Analýza medicínskych snímok
- Detekcia mincí a kruhových objektov

D) Výhody:
- Robustnosť voči šumu
- Detekcia prerušovaných čiar
- Možnosť detekcie viacerých objektov naraz

E) Nevýhody:
- Výpočtová náročnosť
- Pamäťová náročnosť pre 3D akumulátor
- Citlivosť na voľbu parametrov

Príklad kompletného použitia:
```python
def detect_lines_and_circles(image):
    # Predspracovanie
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detekcia priamok
    lines_accumulator, rhos, thetas = hough_transform_lines(edges)
    lines = find_peaks_in_accumulator(lines_accumulator, threshold=100)
    
    # Detekcia kružníc
    circles_accumulator = hough_transform_circles(edges, r_min=20, r_max=100)
    circles = find_peaks_in_3d_accumulator(circles_accumulator, threshold=50)
    
    return lines, circles
```

Tieto poznatky sú dôležité pre skúšku a praktické použitie v počítačovom videní. Odporúčam porozumieť:
- Princípu transformácie priestorov
- Rozdielu medzi detekciou priamok a kružníc
- Optimalizáciám a praktickým aspektom implementácie
- Možnostiam využitia v reálnych aplikáciách

14 OTAZKA: Dôvody analýzy pohybu v obraze, predpoklady urýchľujúce analýzu pohybu, diferenciálna metóda, kumulatívny diferenčný obraz, modelovanie pozadia, motion field a optický tok (v čom je rozdiel), optický tok pomocou sledovania čŕt.

Na základe pridanej prednášky pripravím aktualizovanú a presnú odpoveď:

1. DÔVODY ANALÝZY POHYBU V OBRAZE
- Detekcia pohybu v bezpečnostných systémoch
- Sledovanie objektov (object tracking)
- Segmentácia pohybujúcich sa objektov
- Rekonštrukcia 3D tvaru z pohybu (structure from motion)
- Rozšírená realita (augmented reality)
- Riadenie dopravy
- Medicínske zobrazovanie
- Kompresia videa

2. PREDPOKLADY URÝCHĽUJÚCE ANALÝZU POHYBU
- Jas každého pixelu je konštantný v čase (Brightness constancy)
- Body v okolí pohybujúceho sa pixelu sa pohybujú podobne (Spatial coherence)
- Scéna môže byť:
  - Statická kamera, pohybujúce sa objekty
  - Statická scéna, pohybujúca sa kamera
  - Kombinácia oboch

3. DIFERENCIÁLNA METÓDA
- Založená na rozdieloch medzi po sebe nasledujúcimi snímkami
- Detekuje zmeny na úrovni pixelov
- Často používaná v aplikáciách dohľadu
- Modifikácia: diferencia v hranovom obraze
  - Kombinuje diferenčný obraz D(i,j) s obrazom hrán S(i,j)
  - Hrany sa získajú hranových detektorom
  - Použitie operácie AND pre premietnutie hrán

4. KUMULATÍVNY DIFERENČNÝ OBRAZ (ADI - Accumulative Difference Image)
- Akumuluje rozdiely medzi snímkami v čase
- Zachytáva históriu pohybu v scéne
- Užitočný pre analýzu dlhodobejších zmien

5. MODELOVANIE POZADIA
Algoritmus pomocou mediánového filtra:
1. Inicializácia: Snímanie K snímkov, výpočet mediánovej intenzity pre každý pixel
2. Snímanie ďalšieho obrazu a výpočet rozdielu s mediánovou hodnotou
3. Prahovanie pre minimalizáciu šumu
4. Filtrovanie a morfologické operácie pre odstránenie šumu
5. Prepočítanie mediánu so zahrnutím nových pixelov pozadia
6. Opakovanie procesu

6. MOTION FIELD VS OPTICKÝ TOK

Motion Field:
- Ideálna reprezentácia 3D pohybu premietnutého do roviny kamery
- Závisí na konkrétnych bodoch v priestore
- Časová derivácia pozície všetkých bodov obrazu zodpovedajúcich fixným 3D bodom

Optický tok (Optical Flow):
- Založený na porovnávaní zmien jasu v krátkom časovom intervale
- Závisí na zmene jasu v obraze
- Podmienky:
  - Konštantný jas pixelu v čase
  - Podobný pohyb bodov v okolí

Hlavný rozdiel:
- Motion field závisí na skutočnom pohybe bodov
- Optický tok závisí na zmenách jasu v obraze
- Nemusia byť vždy totožné (napríklad pri rotujúcej guli s jednotnou textúrou)

7. OPTICKÝ TOK POMOCOU SLEDOVANIA ČŔT
- Založený na detekcii a sledovaní významných bodov v obraze
- Proces:
  1. Detekcia významných bodov (čŕt)
  2. Sledovanie týchto bodov medzi snímkami
  3. Výpočet vektorov pohybu
- Využitie napríklad v:
  - Medicínskom zobrazovaní (analýza pohybu orgánov)
  - Sledovaní objektov
  - Analýze pohybu v scéne

Praktické aspekty:
- Potreba riešiť problémy so šumom
- Správne nastavenie parametrov detekcie
- Optimalizácia výpočtovej náročnosti
- Riešenie prekrývajúcich sa objektov

15 OTAZKA Vysvetlenie princípu pasívnej stereovízie pre kamery s paralelnými optickými osami, block matching algoritmus pri výpočte disparity, metódy určovania podobnosti blokov pri block matchingu, vzťah disparity a vzdialenosti bodu od kamery. Čo je to epipolárne obmedzenie, epipolárne línie. Problémy pri hľadaní zhody (stereo-korešpondencie) pri pasívnej sterovízii. Príklady metód získania 3D obrazu: aktívna stereovízia, LiDAR, ToF kamera

Na základe prednášky pripravím odpoveď zameranú na stereovíziu a 3D snímanie:

1. ZÍSKANIE TRETIEHO ROZMERU (VŠEOBECNÉ METÓDY)
- Kombinácia dvoch pohľadov (stereo videnie)
- Sériové rezy kolmé na os Z
- Použitie hĺbkomerov (aktívne, pasívne)
- Špeciálne usporiadanie zdroja svetla a snímača
- Kombinácia obrazov z rôznych uhlov (CT)

2. PASÍVNA STEREOVÍZIA S PARALELNÝMI KAMERAMI

A) Princíp:
- Využíva dve kamery s paralelnými optickými osami
- Rozdiel v pozícii objektu na snímkach z dvoch kamier (disparita)
- Čím je objekt bližšie, tým väčšia disparita

B) Block Matching algoritmus:
- Rozdelenie obrazu na bloky
- Hľadanie zodpovedajúcich blokov v druhom obraze
- Výpočet disparity pre každý blok

C) Metódy určovania podobnosti blokov:
- Sum of Absolute Differences (SAD)
- Sum of Squared Differences (SSD)
- Normalized Cross-Correlation (NCC)
- Census transform
- Rank transform

D) Vzťah disparity a vzdialenosti:
- Z = (f × B) / d
  kde: Z - vzdialenosť bodu
       f - ohnisková vzdialenosť
       B - vzdialenosť medzi kamerami (baseline)
       d - disparita

3. EPIPOLÁRNA GEOMETRIA

A) Epipolárne obmedzenie:
- Zjednodušuje hľadanie korešpondencií
- Bod v jednom obraze leží na epipolárnej línii v druhom obraze

B) Epipolárne línie:
- Priesečníky epipolárnej roviny s obrazovými rovinami
- Pri paralelných kamerách sú horizontálne
- Redukujú 2D problém hľadania na 1D

4. PROBLÉMY PRI HĽADANÍ STEREO-KOREŠPONDENCIE

A) Základné problémy:
- Oklúzie (časti viditeľné len z jednej kamery)
- Opakujúce sa vzory
- Homogénne oblasti
- Perspektívne skreslenie
- Rozdielne osvetlenie v obrazoch

5. ALTERNATÍVNE METÓDY ZÍSKANIA 3D OBRAZU

A) Aktívna stereovízia:
- Projekcia známeho vzoru (structured light)
- Analýza deformácie vzoru na povrchu
- Príklad: Kinect (kombinuje structured light s:
  - depth from focus
  - depth from stereo)

B) LiDAR (Light Detection and Ranging):
- Meranie vzdialenosti pomocou laseru
- Analýza odrazeného svetla
- Možnosť 2D alebo 3D skenovania
- Využitie: autonómne vozidlá, mapovanie terénu

C) ToF (Time of Flight) kamera:
- Meranie času letu svetelného pulzu
- Celá scéna zachytená naraz (scannerless)
- Príklad: Intel RealSense
- Princíp:
  1. Vyslanie krátkeho svetelného pulzu
  2. Meranie času návratu odrazeného svetla
  3. Výpočet vzdialenosti na základe oneskorenia

PRAKTICKÉ ASPEKTY:
- Kalibrácia kamier
- Spracovanie šumu
- Výpočtová náročnosť
- Presnosť merania
- Vplyv okolitého osvetlenia

Dôležité pre skúšku:
1. Pochopiť princíp výpočtu disparity
2. Vedieť vysvetliť epipolárne obmedzenie
3. Poznať problémy stereo-korešpondencie
4. Rozumieť rozdielom medzi aktívnymi a pasívnymi metódami
5. Vedieť vysvetliť princípy rôznych 3D zobrazovacích metód

16 OTAZKA Textúry, charakteristika a rozdelenie, škálová závislosť, textúrny element, textúrny gradient. Štruktúrny, fourierovský a štatistický prístup ku analýze textúr, v čom spočívajú. Lawsove textúrne masky. Výpočet kookurenčnej matice, kritériá odvodené z kookurenčnej matice a ich použitie na klasifikáciu textúr, príklad klasifikácie textúr s využitím zhlukovania.  

Let me help explain textures and their analysis based on the provided lecture materials. I'll break this down systematically:

1. Texture Characteristics and Definition:
- Texture refers to the surface characteristics defined by lay, surface roughness, and waviness
- In image processing, texture provides information about the spatial arrangement of colors or intensities in an image
- Texture gradient is an important concept where texture appears finer as distance increases (like looking at grass from different distances)

2. Main Approaches to Texture Analysis:

A. Structural Approach:
- Views texture as a set of primitive texels (texture elements) in regular or repeated relationships
- Focuses on identifying basic texture patterns and their arrangements

B. Fourier Approach:
- Analyzes local Fourier transforms of the image
- Provides information about:
  - Spatial frequencies contributions
  - Dominant orientations in the texture
- Uses power spectrum analysis with:
  - Ring filters for frequency band contributions
  - Wedge filters for edge orientation information

C. Statistical Approach:
- Characterizes texture using statistical measures from grayscale intensities
- Key method: Co-occurrence Matrices
  - Process:
    1. Separate intensity into discrete levels
    2. Choose displacement vector
    3. Count pixel pairs with specific gray levels
    4. Normalize the matrix
  - Six important characteristics can be derived from these matrices (though specific characteristics aren't listed in the document)

3. Practical Applications:

A. Texture Segmentation:
- Uses texture characteristics to detect regions and contours
- Can employ clustering techniques for classification
- Example shown using graycomatrix function for analyzing horizontal pixel relationships

4. Advanced Concepts:

- Texture Gradient: Used for surface orientation estimation
- Scale Dependency: Texture appearance changes with viewing distance
- Edge Density and Direction: Used as initial steps in texture analysis
  - Counts edge pixels in fixed regions
  - Analyzes edge directions for texture characterization

The document mentions classification using clustering but doesn't specifically detail Laws' texture masks, though these are typically used for texture feature extraction.

For texture classification using clustering, the document shows:
1. Feature selection from texture characteristics
2. Manual parameter setting
3. Cluster center determination
4. Winner-takes-all clustering approach

Would you like me to elaborate on any of these aspects or focus on a specific part of texture analysis?

17 OTAZKA Vysvetlite pojmy ohnisková vzdialenosť objektívu, clonové číslo, rýchlosť uzávierky, hĺbka ostrosti. Ako vplýva na hĺbku ostrosti nastavenie clony a ohnisková vzdialenosť objektívu? Aký vplyv má na obraz použitie objektívov s rôznou ohniskovou vzdialenosťou. Čo predstavuje správna expozícia snímky, čo nás pri správnej expozícii obmedzuje? 

Na základe prednášky "Basics of Photography" a otázky, ktorá pokrýva základné pojmy a ich vzťah k fotografii, ponúkam podrobné vysvetlenie:

---

### **1. Ohnisková vzdialenosť objektívu**
Ohnisková vzdialenosť je miera schopnosti optického systému sústreďovať (konvergovať) alebo rozptyľovať (divergovať) svetlo. Vyjadruje sa v milimetroch (mm) a určuje, aký uhol záberu (zorné pole) objektív zachytáva. 
- **Kratšia ohnisková vzdialenosť (širokouhlý objektív)**: Väčší uhol záberu, lepšia hĺbka ostrosti, vhodný na krajiny alebo architektúru. 
- **Dlhšia ohnisková vzdialenosť (teleobjektív)**: Užší uhol záberu, zväčšenie vzdialených objektov, plytká hĺbka ostrosti, vhodný na šport alebo portréty.

---

### **2. Clonové číslo**
Clonové číslo (f/číslo) predstavuje pomer ohniskovej vzdialenosti k priemeru vstupnej pupily (clony). 
- **Nižšie f/číslo** (väčší otvor): Viac svetla, plytšia hĺbka ostrosti, použiteľné pri slabom osvetlení.
- **Vyššie f/číslo** (menší otvor): Menej svetla, väčšia hĺbka ostrosti, vhodné na krajiny alebo makrofotografiu.

---

### **3. Rýchlosť uzávierky**
Rýchlosť uzávierky určuje, ako dlho zostane svetlo dopadať na senzor alebo film. 
- **Krátka rýchlosť (napr. 1/1000 s)**: Zachytáva rýchly pohyb, minimalizuje rozmazanie.
- **Dlhá rýchlosť (napr. 1/10 s)**: Zachytáva pohyb vo forme rozmazania, vhodná na kreatívne efekty (napr. rozmazanie vody).

---

### **4. Hĺbka ostrosti (DOF)**
Hĺbka ostrosti je rozsah vzdialeností, kde sú objekty v zábere ostré. 
- **Faktory ovplyvňujúce hĺbku ostrosti**:
  - **Clonové číslo**: Väčšie clonové číslo (menšia clona) zvyšuje hĺbku ostrosti.
  - **Ohnisková vzdialenosť**: Dlhšia ohnisková vzdialenosť znižuje hĺbku ostrosti.
  - **Vzdialenosť od objektu**: Čím bližšie je objekt, tým je hĺbka ostrosti menšia.

---

### **5. Vplyv objektívov s rôznou ohniskovou vzdialenosťou na obraz**
- **Širokouhlé objektívy**: Zvýrazňujú vzdialenosti medzi objektmi, poskytujú väčšiu hĺbku ostrosti, vhodné na krajiny alebo interiéry.
- **Teleobjektívy**: Komprimujú vzdialenosti, vytvárajú ilúziu bližšieho vzťahu medzi objektmi, vhodné na portréty alebo detailné snímky.

---

### **6. Správna expozícia snímky**
Správna expozícia znamená, že snímka obsahuje dostatok detailov v tieňoch aj vo svetlých častiach bez straty informácií. Dosahuje sa kombináciou:
- **Clonového čísla**, 
- **Rýchlosti uzávierky**, 
- **ISO** (citlivosť senzoru na svetlo).

**Obmedzenia pri expozícii**:
- **Clona**: Obmedzená vlastnosťami objektívu (maximálne a minimálne f/číslo).
- **Uzávierka**: Pri dlhých časoch je potrebný statív, aby sa predišlo rozmazaniu.
- **ISO**: Vyššie ISO pridáva šum.

---

### **Záver**
Fotografické techniky, ako práca s ohniskovou vzdialenosťou, clonovým číslom a rýchlosťou uzávierky, zásadne ovplyvňujú vzhľad snímky. Hlbšie pochopenie týchto faktorov je kľúčom k vytváraniu kreatívnych a technicky správnych fotografií.

18 OTAZKA Čo ovplyvňuje ISO nastavenie, aký má vplyv na šum v obraze? Čo to je dynamický rozsah senzoru, ako sa líši od ľudského oka. Čo je to HDR composite. Čo to je focus stacking? Ako možno redukovať šum pri fotografovaní statických objektov? Čo ovplyvňuje množstvo detailov zachytiteľných digitálnou kamerou? (acutance vs. resolution) Čo je to chromatická aberácia a prečo vzniká?

Na základe otázky poskytujem detailné vysvetlenie k jednotlivým bodom:

---

### **1. ISO nastavenie a jeho vplyv na šum**
- **ISO** (International Standards Organization) určuje citlivosť senzora na svetlo. 
  - **Nižšie ISO (napr. 100–200)**: Menšia citlivosť, menej šumu, vhodné pri dobrom osvetlení.
  - **Vyššie ISO (napr. 1600 a viac)**: Väčšia citlivosť, viac šumu, vhodné pri slabom osvetlení.
- **Šum** vzniká, keď sa zosilňuje signál zo senzora, pričom sa zosilňujú aj nežiadúce elektronické poruchy (noise).

---

### **2. Dynamický rozsah senzoru a jeho rozdiel od ľudského oka**
- **Dynamický rozsah senzoru**: Rozsah medzi najtmavším a najsvetlejším bodom, ktoré senzor dokáže zachytiť bez straty detailov.
  - **Bežný rozsah senzoru**: ~10–15 expozičných krokov (stops).
  - **Ľudské oko**: Má širší dynamický rozsah (~20 stops) a dokáže sa dynamicky prispôsobiť rôznym úrovniam jasu, čo fotoaparáty nedokážu bez dodatočných techník (napr. HDR).

---

### **3. HDR composite**
- **HDR (High Dynamic Range) composite**: Technika spájania viacerých snímok s rôznymi expozíciami (napr. podexponovaná, správne exponovaná, preexponovaná). 
  - Výsledkom je obraz s väčším dynamickým rozsahom, kde sú zachované detaily v tieňoch aj svetlých oblastiach.
  - Používa sa na scény s vysokým kontrastom, ako napríklad pri fotografovaní krajiny za jasného slnečného dňa.

---

### **4. Focus stacking**
- **Focus stacking**: Digitálna technika kombinujúca viac snímok s rôznymi zaostrenými rovinami (hlboká ostrosť každého záberu na inú časť scény).
  - Výsledkom je obraz s väčšou hĺbkou ostrosti než akýkoľvek jednotlivý zdrojový obrázok.
  - Používa sa pri makrofotografii alebo pri fotografovaní produktov, kde je dôležitá ostrá detailnosť v celom zábere.

---

### **5. Redukcia šumu pri fotografovaní statických objektov**
- **Redukcia šumu**:
  - Použitie **nižšieho ISO** (napr. ISO 100–200).
  - **Dlhšie expozičné časy** s použitím statívu.
  - Použitie techniky **image stacking**: Viaceré snímky sú spriemerované, čo redukuje náhodný šum.
  - **Post-procesing**: Softvérové nástroje ako Adobe Lightroom na redukciu šumu.

---

### **6. Množstvo detailov zachytiteľných kamerou (acutance vs. resolution)**
- **Resolution (rozlíšenie)**: Počet zachytených pixelov, teda schopnosť odlíšiť blízko umiestnené detaily. Vyššie rozlíšenie znamená viac pixelov a jemnejšie detaily.
- **Acutance (ostrosť)**: Subjektívne vnímanie ostrosti obrazu, ovplyvnené rýchlosťou prechodu medzi svetlom a tieňom. Väčší kontrast hrán môže zvýšiť vnímanú ostrosť, aj keď rozlíšenie ostáva rovnaké.

---

### **7. Chromatická aberácia**
- **Definícia**: Optická chyba objektívu spôsobená tým, že rôzne vlnové dĺžky svetla (farby) sa lámu pod odlišnými uhlami.
- **Prečo vzniká**: Svetlo prechádzajúce cez šošovku sa rozkladá na jednotlivé farby (podobne ako pri prizme). Tieto farby sa nezhromažďujú do jedného bodu, čo vedie k farebným okrajom (napr. purpurové alebo zelené fringing).
- **Ako ju znížiť**:
  - Použitie objektívov s **asférickými šošovkami** alebo špeciálnymi nízko-dispersnými materiálmi.
  - **Clonenie**: Zníženie otvoru clony môže redukovať chromatickú aberáciu.
  - **Post-procesing**: Softvér na úpravu fotografií (napr. Lightroom alebo Photoshop) umožňuje odstránenie chromatickej aberácie.

---

### **Záver**
Pochopenie týchto faktorov a techník umožňuje fotografom efektívne ovládať kvalitu snímok, minimalizovať chyby a maximalizovať technickú i umeleckú hodnotu výsledných fotografií.



