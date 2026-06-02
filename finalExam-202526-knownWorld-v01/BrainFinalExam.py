# Entrega de Control de Robotica y Percepcion Computacional - Examen Final.
# Alumnas:
#   Lucia Fuentes Gonzalez
#   Miriam Bernat Jimenez
#   Tania Mobasser Aslfakouri
#
# Brain que COMBINA las tres practicas:
#   - BrainFollowLine.py            -> bucle del Brain, control PD, evasion con sonar,
#                                      busqueda de linea perdida.
#   - 02_analisis_escena.ipynb      -> segmentacion por color (NearestCentroid),
#                                      ROI/endpoints, clasificacion de escena,
#                                      flecha -> salida elegida en cruces,
#                                      clasificador de marcas (Hu + KNN).
#   - 03_segmentacion_circulos.ipynb-> deteccion de circulo y distancia (pinhole).
#
# Los clasificadores se reimplementan en numpy puro (sin sklearn) para que el
# Brain sea autocontenido en la VM del simulador (solo necesita cv2 + numpy).

from pyrobot.brain import Brain

import os
import numpy as np
import cv2


# =============================================================================
# CONFIGURACION
# =============================================================================

# getImage() de pyrobot devuelve BGR (por eso el Brain original hace BGR2GRAY).
# segmentar() espera RGB (igual que iio.imread en el notebook). Si los colores
# salieran invertidos (la linea se clasifica como marca, etc.), pon False.
CAMERA_RETURNS_BGR = True

# Fuente de imagen:
#   True  -> camara del robot via pyrobot (self.robot.getImage()). Usar en el
#            SIMULADOR Stage.
#   False -> camara USB via OpenCV (cv2.VideoCapture). Usar en el ROBOT FISICO:
#            la Logitech C920 es un webcam aparte, no la gestiona el driver Aria.
USE_ROBOT_CAMERA = True
CAMERA_INDEX = 0       # indice de la camara USB (0 = la primera) si USE_ROBOT_CAMERA=False
# La C920 captura en 16:9 (640x360 es su resolucion mas baja); 320x240 (4:3) del
# simulador no es nativo. Estos valores solo aplican a la camara USB.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360

# Ancho al que se reescala el frame ANTES de procesarlo. Mantiene la CPU ligera
# (el PC del robot es modesto) y conserva validos los umbrales en pixeles que se
# afinaron a ~320 px. El simulador ya viene a 320, asi que no le afecta.
PROC_WIDTH = 320

# Mostrar ventanas de depuracion con opencv. IMPORTANTE: en el robot fisico,
# conectado por 'ssh -X', cada imshow reenvia el frame por red y RALENTIZA mucho
# el bucle de control. Ponlo a False al ejecutar en el robot.
DEBUG_VIEW = True

# Solo imprimir las lineas de marcas detectadas (MARCA |). Silencia FOLLOW,
# CRUCE, SEARCH y AVOID. Util cuando solo interesa el resultado de clasificacion.
PRINT_ONLY_MARCA = False

# Activar deteccion de circulo + distancia (practica 03). En este mundo conocido
# no hay objeto circular, asi que por defecto esta desactivado. Ponlo a True si
# usas un mundo con un circulo/pelota.
ENABLE_CIRCLE = False

# Centroides del NearestCentroid entrenado en 02_analisis_escena.ipynb,
# en espacio rg-normalizado. Orden: marca(rojo), fondo, linea(azul).
CENTROIDES = np.array([
    [0.49497286, 0.28406798],   # 0 = marca (rojo)
    [0.33962690, 0.36875265],   # 1 = fondo
    [0.15409867, 0.32765910],   # 2 = linea (azul)
], dtype=np.float64)
CLASE_MARCA, CLASE_FONDO, CLASE_LINEA = 0, 1, 2

# Carpeta con el dataset de marcas (subcarpetas = clases). Se prueban varias
# rutas relativas al Brain; copia images/marcas junto al examen en la VM, o
# define la variable de entorno MARCAS_DIR.
_AQUI = os.path.dirname(os.path.abspath(__file__))
MARCAS_DIR_CANDIDATAS = [
    os.environ.get('MARCAS_DIR', ''),
    os.path.join(_AQUI, 'images', 'marcas'),
    os.path.join(_AQUI, '..', 'images', 'marcas'),
    os.path.join(_AQUI, '..', '..', 'images', 'marcas'),
]
EXTS_IMG = {'.png', '.jpg', '.jpeg', '.bmp'}

# Parametros del modelo pinhole para el circulo (practica 03).
FOCAL_PX = 360.0
DIAMETRO_REAL_MM = 65.0
MIN_CIRCULARIDAD = 0.70
MIN_AREA_PX_CIRC = 500


# =============================================================================
# PERCEPCION  (funciones portadas de 02_analisis_escena.ipynb)
# =============================================================================

def normaliza_rg(im):
    """Normaliza RGB por la suma de canales (cromaticidad). Devuelve float (H,W,3)."""
    im_float = im.astype(np.float64)
    suma = im_float.sum(axis=2, keepdims=True)
    suma[suma == 0] = 1.0
    return im_float / suma


def segmentar(im_rgb):
    """NearestCentroid en numpy: clasifica cada pixel por su (r,g) normalizado.
    Devuelve mapa de etiquetas (H,W) con 0=marca, 1=fondo, 2=linea."""
    H, W, _ = im_rgb.shape
    im_norm = normaliza_rg(im_rgb)
    pixels = im_norm[:, :, :2].reshape(-1, 2)              # (N, 2)
    # distancia a cada centroide -> (N, 3); etiqueta = centroide mas cercano
    d = ((pixels[:, None, :] - CENTROIDES[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).reshape(H, W)


def mascaras(etiquetas):
    """Devuelve (mascara_linea, mascara_marca) booleanas."""
    return (etiquetas == CLASE_LINEA), (etiquetas == CLASE_MARCA)


def get_roi(shape, franja=0.75, anticipacion=0.0):
    """Franja horizontal central (75% del alto por defecto)."""
    H, W = shape[:2]
    alto = int(H * franja)
    centro = H // 2 - int(H * anticipacion)
    y0 = max(0, centro - alto // 2)
    y1 = min(H, y0 + alto)
    return y0, y1, 0, W


def aplicar_roi(arr, roi):
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1]


def runs_in_border(border_pixels):
    """Lista de (i_inicio, i_fin) de los tramos True en un vector binario 1D."""
    runs = []
    n = len(border_pixels)
    i = 0
    while i < n:
        if border_pixels[i]:
            j = i
            while j < n and border_pixels[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def endpoints_en_roi(m_linea, roi, grosor=4):
    """Endpoints donde la linea cruza los 4 bordes de la ROI.
    Devuelve lista de dicts {'side', 'pt':(x,y), 'role':'entrada|salida'}.

    Cada borde se mira como una BANDA de `grosor` pixeles (no una sola fila):
    basta con que haya pixel de linea en CUALQUIER fila/columna de la banda para
    contar el cruce. Asi la segmentacion no necesita llegar al pixel exacto del
    borde para detectar el endpoint."""
    y0, y1, x0, x1 = roi
    sub = m_linea[y0:y1, x0:x1]
    H, W = sub.shape
    g = max(1, min(grosor, H // 2, W // 2))

    borders = [
        ('bottom', sub[H - g:, :].any(axis=0), lambda m: (x0 + m, y0 + H - 1), 'entrada'),
        ('top',    sub[:g, :].any(axis=0),      lambda m: (x0 + m, y0),          'salida'),
        ('left',   sub[:, :g].any(axis=1),      lambda m: (x0, y0 + m),          'salida'),
        ('right',  sub[:, W - g:].any(axis=1),  lambda m: (x0 + W - 1, y0 + m), 'salida'),
    ]

    endpoints = []
    for side, pixels, to_pt, role in borders:
        for a, b in runs_in_border(pixels):
            endpoints.append({'side': side, 'pt': to_pt((a + b) // 2), 'role': role})

    # Si no hay entrada por abajo, el endpoint mas inferior pasa a ser entrada.
    if not any(e['role'] == 'entrada' for e in endpoints) and endpoints:
        max(endpoints, key=lambda e: e['pt'][1])['role'] = 'entrada'

    return endpoints


def clasifica_escena(endpoints, umbral_curvatura=10):
    """Etiqueta de escena segun numero de endpoints y geometria."""
    n = len(endpoints)
    if n >= 4:
        return 'cruce 3 salidas'
    if n == 3:
        return 'cruce 2 salidas'
    if n < 2:
        return 'escena no clara'

    e_in = next((e for e in endpoints if e['role'] == 'entrada'), None)
    e_out = next((e for e in endpoints if e['role'] == 'salida'), None)
    if e_in is None or e_out is None:
        return 'linea recta'

    dx_ie = e_out['pt'][0] - e_in['pt'][0]
    if abs(dx_ie) <= umbral_curvatura:
        return 'linea recta'
    return 'curva izda' if dx_ie < 0 else 'curva dcha'


def mayor_blob(mask_bool):
    """Mascara booleana con SOLO la mayor componente conexa (para aislar la
    flecha de otras manchas rojas que pueda haber en la ROI)."""
    m = mask_bool.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_bool
    # componente 0 = fondo; elegir la de mayor area
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def orientacion_flecha(m_marca_roi, area_min=30):
    """Orientacion de la flecha por tensor de inercia (algoritmo del notebook).
    Devuelve (cx, cy, tx, ty) en coords de la ROI, o None."""
    A = m_marca_roi.astype(np.float64)
    total = A.sum()
    if total < area_min:
        return None

    rows, cols = A.shape
    XX, YY = np.meshgrid(np.arange(cols, dtype=np.float64),
                         np.arange(rows, dtype=np.float64))

    cx = (XX * A).sum() / total
    cy = (YY * A).sum() / total
    dx = XX - cx
    dy = YY - cy

    Ix = (dy ** 2 * A).sum() / total
    Iy = (dx ** 2 * A).sum() / total
    Ixy = (dx * dy * A).sum() / total

    sita = 0.0
    for _ in range(1000):
        tangent = -np.sin(2 * sita) * Ix + np.sin(2 * sita) * Iy - 2 * np.cos(2 * sita) * Ixy
        sita -= 0.02 * tangent

    proj_main = (np.cos(sita) * dx + np.sin(sita) * dy) * A / total
    I_p = (proj_main[proj_main > 0] ** 2).sum()
    I_n = (proj_main[proj_main < 0] ** 2).sum()

    proj_perp = (np.cos(sita + np.pi / 2) * dx + np.sin(sita + np.pi / 2) * dy) * A / total
    O_p = (proj_perp[proj_perp > 0] ** 2).sum()
    O_n = (proj_perp[proj_perp < 0] ** 2).sum()

    if abs(I_p - I_n) >= abs(O_p - O_n):
        angle = sita if I_p < I_n else sita + np.pi
    else:
        angle = (sita + np.pi / 2) if O_p < O_n else (sita + 3 * np.pi / 2)

    py_idx, px_idx = np.where(A > 0)
    proj_tip = (px_idx - cx) * np.cos(angle) + (py_idx - cy) * np.sin(angle)
    tip = np.argmax(proj_tip)
    return float(cx), float(cy), float(px_idx[tip]), float(py_idx[tip])


def salida_de_label(endpoints, label, xc):
    """Mapea una etiqueta de direccion ('left'/'straight'/'right') a la salida
    concreta entre los endpoints actuales. Devuelve el dict de salida o None."""
    salidas = [e for e in endpoints if e['role'] == 'salida']
    if not salidas:
        return None
    if label == 'left':
        return min(salidas, key=lambda e: e['pt'][0])       # salida mas a la izda
    if label == 'right':
        return max(salidas, key=lambda e: e['pt'][0])       # salida mas a la dcha
    # 'straight': salida mas centrada, prefiriendo la del borde superior.
    tops = [e for e in salidas if e['side'] == 'top']
    cand = tops if tops else salidas
    return min(cand, key=lambda e: abs(e['pt'][0] - xc))


def salida_por_flecha(arrow_info, endpoints, roi, deadzone_deg=35.0):
    """Decide la salida en un cruce a partir de la inclinacion de la flecha.

    En vez de la similitud de coseno (fragil cuando la flecha se ve parcial),
    clasificamos la flecha en 'left' / 'straight' / 'right' segun cuanto se
    inclina respecto a la vertical de la imagen, con una zona muerta amplia a
    favor de 'recto'. Asi una flecha casi vertical (recta), aunque se vea a
    medias, no se confunde con una salida lateral.

    Devuelve (salida_dict, label) o (None, 'none')."""
    if arrow_info is None:
        return None, 'none'
    cx, cy, tx, ty = arrow_info
    vx, vy = tx - cx, ty - cy
    # Inclinacion respecto a la vertical (0 = vertical/recto, 90 = horizontal).
    # Usamos valores absolutos para ignorar el signo (la punta puede salir hacia
    # arriba o hacia abajo segun el recorte del blob).
    lean = np.degrees(np.arctan2(abs(vx), abs(vy) + 1e-9))
    if lean < deadzone_deg:
        label = 'straight'
    elif vx < 0:
        label = 'left'
    else:
        label = 'right'
    xc = (roi[2] + roi[3]) / 2.0
    return salida_de_label(endpoints, label, xc), label


def error_seguimiento(m_linea, roi, alto_franja=15):
    """Error = x_centro_linea - x_centro_imagen, medido en una franja sobre el
    borde inferior de la ROI (lo mas cercano al robot)."""
    H, W = m_linea.shape
    _, y_roi_bot, _, _ = roi
    y_centro = min(y_roi_bot, H - 1)
    y0 = max(0, y_centro - alto_franja)
    y1 = min(H, y_centro)

    franja = m_linea[y0:y1, :]
    ys, xs = np.where(franja)
    if len(xs) < 5:
        return None, None, W / 2.0
    return xs.mean() - W / 2.0, xs.mean(), W / 2.0


# =============================================================================
# CLASIFICADOR DE MARCAS  (Hu Moments + KNN, de 02_analisis_escena.ipynb #11)
# =============================================================================

def hu_descriptor(img_bin):
    """7 momentos de Hu con transformacion logaritmica estandar."""
    moments = cv2.moments(img_bin, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def extraer_marca_binaria(mask, area_min=50):
    """Imagen 0/255 con solo el mayor contorno relleno, o None."""
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cont = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cont) < area_min:
        return None
    clean = np.zeros_like(m)
    cv2.drawContours(clean, [cont], -1, 255, -1)
    return clean


def normalizar_marca(img_bin, size=100):
    """Recorta al bounding box y redimensiona a size x size."""
    ys, xs = np.where(img_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    recorte = img_bin[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(recorte, (size, size))


# Descriptores de Hu PRECALCULADOS de images/marcas/ (28 muestras, mismo pipeline
# que entrenar()). Embebidos para que el Brain NO necesite el dataset de imagenes
# en el robot: con esto la clasificacion de marcas funciona sin copiar carpetas.
# (Generados con el script _calc_marcas.py.)
MARCAS_CLASES = ['escalera', 'hombre', 'mujer', 'telefono']
MARCAS_Y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
MARCAS_X = [
    [0.550941, 1.459643, 3.603957, 4.866504, 9.164292, 5.778954, 9.198867],
    [0.492817, 1.214445, 3.709520, 4.943960, -9.544864, -6.284238, -9.219567],
    [0.580970, 1.621077, 3.634450, 5.042094, 9.474470, 6.126793, 9.352980],
    [0.550941, 1.459643, 3.603957, 4.866504, 9.164292, 5.778954, 9.198867],
    [0.141581, 0.319415, 3.276124, 3.501798, 6.890662, 3.662622, -8.358397],
    [0.673360, 2.416298, 4.092482, 6.720713, -9.997560, 7.936011, -9.997886],
    [0.287272, 0.649089, 3.358199, 3.719251, 7.258366, 4.052293, -8.380731],
    [0.628221, 1.592848, 3.138117, 3.665661, 7.067748, 4.470364, 8.303023],
    [0.752539, 3.504009, 3.392045, 5.153436, 9.337796, 6.936825, 9.685438],
    [0.724996, 2.283154, 3.309827, 4.611321, 9.205951, -6.756795, -8.564093],
    [0.705546, 2.064926, 3.265584, 4.189338, 7.954536, 5.298323, -8.287297],
    [0.729595, 2.312662, 3.322741, 4.570175, 8.751365, 6.215771, 8.577975],
    [0.726123, 2.229419, 3.347044, 4.505572, 8.593979, 5.993677, 8.541456],
    [0.746171, 2.813379, 3.378800, 5.012056, 9.679176, -7.095173, -9.148463],
    [0.732127, 2.342557, 3.695455, 4.831570, 9.480827, -6.589579, 9.060687],
    [0.704571, 2.070581, 3.605927, 4.617448, 9.199464, -6.653388, 8.723904],
    [0.673911, 1.800562, 3.561516, 4.092926, 7.917183, 5.008698, -9.123862],
    [0.641538, 1.650376, 3.403459, 3.817050, 7.427058, 4.643843, 8.597724],
    [0.593272, 1.448200, 3.294520, 3.643264, 7.113257, 4.378901, -8.164255],
    [0.593046, 1.448204, 3.186147, 3.509136, 6.856565, 4.233335, -8.512201],
    [0.669403, 1.803580, 3.463410, 4.272232, 8.257051, 5.720992, -8.310620],
    [0.474173, 1.264654, 2.115258, 2.663164, 5.248106, 3.800556, 5.165474],
    [0.469153, 1.255406, 2.117151, 2.793181, 5.828677, -4.383963, 5.263880],
    [0.471759, 1.279931, 2.130844, 2.958494, -5.722780, -3.708109, 5.601307],
    [0.567130, 1.772506, 2.341749, 2.771513, 5.498858, 3.873265, 5.460156],
    [0.521650, 1.435924, 2.325260, 3.162607, -6.020409, -3.976777, -6.101045],
    [0.504306, 1.369476, 2.262827, 3.120213, -5.846016, -3.840268, -6.229235],
    [0.550750, 1.598026, 2.373751, 3.200847, -6.052271, -4.063473, -6.284057],
]


class KNNMarcas:
    """KNN (k=3) en numpy puro sobre descriptores de Hu. Se entrena cargando
    images/marcas/<clase>/*.png, o desde los descriptores embebidos (MARCAS_X)
    para no depender del dataset de imagenes en el robot."""

    def __init__(self, k=3):
        self.k = k
        self.X = None          # (N, 7)
        self.y = None          # (N,)
        self.clases = []
        self.ok = False

    def entrenar(self, dir_marcas):
        if not dir_marcas or not os.path.isdir(dir_marcas):
            return False
        clase_dirs = sorted(d for d in os.listdir(dir_marcas)
                            if os.path.isdir(os.path.join(dir_marcas, d)))
        if not clase_dirs:
            return False
        self.clases = clase_dirs
        X, y = [], []
        for idx, clase in enumerate(clase_dirs):
            cdir = os.path.join(dir_marcas, clase)
            for fname in sorted(os.listdir(cdir)):
                if os.path.splitext(fname)[1].lower() not in EXTS_IMG:
                    continue
                img = cv2.imread(os.path.join(cdir, fname))   # BGR
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                etiq = segmentar(rgb)
                mask = (etiq == CLASE_MARCA).astype(np.uint8) * 255
                limpia = extraer_marca_binaria(mask)
                if limpia is not None:
                    limpia = normalizar_marca(limpia)
                if limpia is None:
                    continue
                limpia = cv2.medianBlur(limpia, 5)
                X.append(hu_descriptor(limpia))
                y.append(idx)
        if not X:
            return False
        self.X = np.array(X)
        self.y = np.array(y)
        self.ok = True
        return True

    def cargar_embebido(self):
        """Carga los descriptores precalculados embebidos en el modulo. No
        necesita ningun fichero externo."""
        if not MARCAS_X:
            return False
        self.clases = list(MARCAS_CLASES)
        self.X = np.array(MARCAS_X, dtype=np.float64)
        self.y = np.array(MARCAS_Y, dtype=int)
        self.ok = True
        return True

    def predict(self, descriptor):
        """Devuelve el nombre de clase del voto mayoritario de los k vecinos."""
        if not self.ok:
            return None
        d = np.sqrt(((self.X - descriptor[None, :]) ** 2).sum(axis=1))
        vecinos = self.y[np.argsort(d)[:self.k]]
        clase = np.bincount(vecinos).argmax()
        return self.clases[clase]

# =============================================================================
# BRAIN
# =============================================================================

class BrainFinalExam(Brain):

    NO_FORWARD = 0
    VERY_SLOW_FORWARD = 0.05
    SLOW_FORWARD = 0.1
    MED_FORWARD = 0.5
    FULL_FORWARD = 1.0

    NO_TURN = 0
    MED_LEFT = 0.5
    HARD_LEFT = 1.0
    MED_RIGHT = -0.5
    HARD_RIGHT = -1.0

    OBSTACLE_STOP = 0.40   # caja de frente -> girar fuerte avanzando despacio
    OBSTACLE_WARN = 0.65   # caja al costado -> girar fuerte avanzando medio

    # Ganancias del control PD de seguimiento (como BrainFollowLine).
    LINE_KP = 0.9
    LINE_KD = 0.5

    # Velocidades del controlador visual.
    V_MAX = 0.7
    V_MIN = 0.05

    # Busqueda de linea perdida.
    SEARCH_SLOW_AFTER = 15
    SEARCH_REVERSE_AFTER = 35

    CROSS_COMMIT_STEPS = 22
    CROSS_RELEASE_PX = 45
    STRAIGHT_DEADZONE_DEG = 35

    def setup(self):
        self.last_error = 0.0          # error normalizado [-1, 1]
        self._lost_line_steps = 0
        self._search_dir = self.HARD_RIGHT
        self._last_mark = None

        # Fuente de imagen: camara del robot (simulador) o camara USB (robot
        # fisico). Se usa la USB si se pide explicitamente (USE_ROBOT_CAMERA=False)
        # o si el robot no expone getImage() (p.ej. el AriaRobot del Pioneer, cuya
        # camara C920 es un webcam USB aparte): asi funciona en ambos sin tocar el
        # flag.
        self.capture = None
        use_usb = (not USE_ROBOT_CAMERA) or (not hasattr(self.robot, 'getImage'))
        if use_usb:
            self.capture = cv2.VideoCapture(CAMERA_INDEX)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            if self.capture.isOpened():
                print("[FinalExam] Camara USB abierta (index %d)." % CAMERA_INDEX)
            else:
                print("[FinalExam] AVISO: no se pudo abrir la camara USB index", CAMERA_INDEX,
                      "- prueba otro CAMERA_INDEX (1, 2, ...).")
        else:
            print("[FinalExam] Usando la camara del robot (getImage).")

        # Memoria del cruce en curso.
        self._cross_steps = 0          # pasos restantes de compromiso (0 = inactivo)
        self._cross_side = None        # lado de la salida elegida ('top'/'left'/'right')
        self._cross_err = 0.0          # ultimo error conocido hacia esa salida (px)
        self._cross_best_area = 0      # mayor area de flecha vista en este cruce
        self._cross_label = 'none'     # 'left'/'straight'/'right' decidido

        # Clasificador de marcas: si hay carpeta images/marcas/ se entrena de ella
        # (util para re-entrenar con fotos reales); si no, usa los descriptores
        # EMBEBIDOS, asi no hace falta copiar el dataset al robot.
        self.knn = KNNMarcas(k=3)
        entrenado = False
        for cand in MARCAS_DIR_CANDIDATAS:
            if self.knn.entrenar(cand):
                print("[FinalExam] Marcas entrenadas desde:", os.path.abspath(cand),
                      "->", self.knn.clases)
                entrenado = True
                break
        if not entrenado:
            self.knn.cargar_embebido()
            print("[FinalExam] Marcas: usando descriptores embebidos (%d muestras, %s)."
                  % (len(self.knn.y), self.knn.clases))

    def destroy(self):
        if getattr(self, 'capture', None) is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    def _get_image(self):
        """Captura un frame BGR de la fuente configurada (camara USB del robot
        fisico o camara del robot via pyrobot en el simulador) y lo reescala a
        PROC_WIDTH para aligerar el procesamiento."""
        if self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                return None
        else:
            frame = self.robot.getImage()
        if frame is not None and PROC_WIDTH and frame.shape[1] > PROC_WIDTH:
            h = int(frame.shape[0] * PROC_WIDTH / float(frame.shape[1]))
            frame = cv2.resize(frame, (PROC_WIDTH, h))
        return frame

    # --- sonar -----------------------------------------------------------
    def _min_range(self, group_name, default=3.0):
        try:
            sensors = self.robot.range[group_name]
        except Exception:
            return default
        if not sensors:
            return default
        distances = [s.distance() for s in sensors if s is not None]
        return min(distances) if distances else default

    def _avoid_obstacle(self):
        front = self._min_range("front")
        front_left = self._min_range("front-left")
        front_right = self._min_range("front-right")

        # Caja de frente: girar FUERTE hacia el lado mas libre, avanzando despacio.
        if front < self.OBSTACLE_STOP:
            if front_left < front_right:
                self.move(self.SLOW_FORWARD, self.HARD_RIGHT)
            else:
                self.move(self.SLOW_FORWARD, self.HARD_LEFT)
            if not PRINT_ONLY_MARCA:
                print("AVOID | front=%.2f gira fuerte" % front)
            return True

        # Caja al costado: seguir avanzando pero girando fuerte para apartarse.
        if front_left < self.OBSTACLE_WARN:
            self.move(self.MED_FORWARD, self.HARD_RIGHT)
            if not PRINT_ONLY_MARCA:
                print("AVOID | front_left=%.2f aparta dcha" % front_left)
            return True
        if front_right < self.OBSTACLE_WARN:
            self.move(self.MED_FORWARD, self.HARD_LEFT)
            if not PRINT_ONLY_MARCA:
                print("AVOID | front_right=%.2f aparta izda" % front_right)
            return True
        return False

    # --- control PD ------------------------------------------------------
    def _control_pd(self, error_px, W):
        """error_px -> (forward, turn). error_px>0 = linea/salida a la derecha."""
        e = max(-1.0, min(1.0, error_px / (W / 2.0)))
        de = e - self.last_error
        turn = -(self.LINE_KP * e + self.LINE_KD * de)
        turn = max(self.HARD_RIGHT, min(self.HARD_LEFT, turn))
        self.last_error = e
        # La velocidad baja cuando el giro es grande (igual que BrainFollowLine).
        forward = max(self.V_MIN, self.V_MAX - abs(turn) * 1.5 * self.V_MAX)
        return forward, turn

    def _buscar_linea(self):
        """Comportamiento de busqueda cuando se pierde la linea."""
        self._lost_line_steps += 1
        if self._lost_line_steps == 1:
            if self.last_error > 0:
                self._search_dir = self.MED_RIGHT
            elif self.last_error < 0:
                self._search_dir = self.MED_LEFT

        if self._lost_line_steps < self.SEARCH_SLOW_AFTER:
            self.move(self.SLOW_FORWARD, self._search_dir)
        elif self._lost_line_steps < self.SEARCH_REVERSE_AFTER:
            self.move(self.VERY_SLOW_FORWARD, self._search_dir)
        else:
            self._search_dir *= -1
            self._lost_line_steps = 0
            self.move(self.VERY_SLOW_FORWARD, self._search_dir)
        if not PRINT_ONLY_MARCA:
            print("SEARCH | step=%d last_error=%.3f dir=%.2f"
                  % (self._lost_line_steps, self.last_error, self._search_dir))

    # --- clasificacion de marca -----------------------------------------
    def _clasificar_marca(self, m_marca_roi):
        if not self.knn.ok:
            return None
        mask = m_marca_roi.astype(np.uint8) * 255
        limpia = extraer_marca_binaria(mask)
        if limpia is None:
            return None
        norm = normalizar_marca(limpia)
        if norm is None:
            return None
        norm = cv2.medianBlur(norm, 5)
        return self.knn.predict(hu_descriptor(norm))

    # --- bucle principal -------------------------------------------------
    def step(self):
        cv_image = self._get_image()
        if cv_image is None:                 # sin frame (camara no lista): parar y reintentar
            self.move(self.NO_FORWARD, self.NO_TURN)
            return
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) if CAMERA_RETURNS_BGR else cv_image

        roi = get_roi(rgb.shape)
        etiquetas = segmentar(rgb)
        m_linea, m_marca = mascaras(etiquetas)
        endpoints = endpoints_en_roi(m_linea, roi)
        escena = clasifica_escena(endpoints)
        err_px, x_linea, x_centro = error_seguimiento(m_linea, roi)

        m_marca_roi = aplicar_roi(m_marca, roi)
        W = rgb.shape[1]
        arrow_info, salida = None, None

        if self._avoid_obstacle():
            self._cross_steps = 0
            self._cross_label = 'none'
            if DEBUG_VIEW:
                self._mostrar(cv_image, roi, (endpoints, escena, arrow_info, salida))
            return

        if 'cruce' in escena and m_marca_roi.any():
            blob = mayor_blob(m_marca_roi)
            arrow_info = orientacion_flecha(blob)
            salida, label = salida_por_flecha(arrow_info, endpoints, roi,
                                              self.STRAIGHT_DEADZONE_DEG)
            if salida is not None:
                area = int(blob.sum())
                if self._cross_steps == 0:          # cruce nuevo: reinicia la mejor vista
                    self._cross_best_area = 0
                if area >= self._cross_best_area:    # solo la mejor vista fija la direccion
                    self._cross_best_area = area
                    self._cross_label = label
                self._cross_side = salida['side']
                self._cross_steps = self.CROSS_COMMIT_STEPS

        # PRIORIDAD 2: resolver el cruce usando la direccion MEMORIZADA.
        if self._cross_steps > 0:
            self._cross_steps -= 1
            sal_mem = salida_de_label(endpoints, self._cross_label, x_centro)
            if sal_mem is not None:
                new_err = sal_mem['pt'][0] - x_centro
                coherente = (self._cross_label == 'straight'
                             or (self._cross_label == 'left' and new_err < 0)
                             or (self._cross_label == 'right' and new_err > 0))
                if coherente:
                    self._cross_err = new_err
                    self._cross_side = sal_mem['side']
            forward, turn = self._control_pd(self._cross_err, W)
            self._lost_line_steps = 0
            self.move(forward, turn)
            if not PRINT_ONLY_MARCA:
                print("CRUCE  | %s side=%s steps=%d err=%.1f v=%.2f w=%.2f"
                      % (self._cross_label, self._cross_side, self._cross_steps,
                         self._cross_err, forward, turn))
            if (escena in ('linea recta', 'curva izda', 'curva dcha')
                    and err_px is not None and abs(err_px) < self.CROSS_RELEASE_PX):
                self._cross_steps = 0
                self._cross_side = None
                self._cross_label = 'none'

        # PRIORIDAD 3: seguir la linea (PD sobre el error de segmentacion).
        elif err_px is not None:
            forward, turn = self._control_pd(err_px, W)
            self._lost_line_steps = 0
            self.move(forward, turn)
            if not PRINT_ONLY_MARCA:
                print("FOLLOW | %s err=%.1f v=%.2f w=%.2f" % (escena, err_px, forward, turn))

            # En recta/curva, una mancha roja lateral es una marca: clasificarla.
            if 'cruce' not in escena:
                marca = self._clasificar_marca(m_marca_roi)
                if marca is not None and marca != self._last_mark:
                    print("MARCA  | detectada: %s" % marca)
                    self._last_mark = marca

        # PRIORIDAD 4: buscar linea
        else:
            self._buscar_linea()

        if DEBUG_VIEW:
            self._mostrar(cv_image, roi, (endpoints, escena, arrow_info, salida))

    # --- depuracion visual ----------------------------------------------
    def _mostrar(self, cv_image, roi, info):
        vis = cv_image.copy()
        if roi is not None and info is not None:
            endpoints, escena, arrow_info, salida = info
            y0, y1, x0, x1 = roi
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
            for e in endpoints:
                color = (0, 255, 0) if e['role'] == 'entrada' else (255, 0, 255)
                cv2.circle(vis, e['pt'], 5, color, -1)
            if salida is not None:
                cv2.circle(vis, salida['pt'], 9, (0, 255, 255), 2)
            # Flecha estimada: centroide -> punta (en coords globales).
            if arrow_info is not None:
                cx, cy, tx, ty = arrow_info
                cv2.arrowedLine(vis, (int(cx + x0), int(cy + y0)),
                                (int(tx + x0), int(ty + y0)),
                                (0, 200, 255), 2, tipLength=0.3)
            cv2.putText(vis, str(escena), (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self._cross_steps > 0:
                cv2.putText(vis, "COMMIT %s->%s (%d)"
                            % (self._cross_label, self._cross_side, self._cross_steps),
                            (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("FinalExam", vis)
        cv2.waitKey(1)


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))
    return BrainFinalExam('BrainFinalExam', engine)
