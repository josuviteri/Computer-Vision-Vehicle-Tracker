import cv2 as cv
import numpy as np

# Cargar la imagen y convertirla a escala de grises
imagen = cv.imread("media/glass-caster-RGB.tif")

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen")
    exit()

grises = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

# Detector SIFT
sift = cv.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

# Detectar puntos clave
kp = sift.detect(grises, None)

# Detectar puntos clave y calcular descriptores
kp, des = sift.detectAndCompute(grises, None)

print(f"Número de puntos clave detectados: {len(kp)}")

# Dibujar puntos clave en modo simple
imagen_kp_simple = cv.drawKeypoints(imagen, kp, None, color=(0, 255, 0), 
                                flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

# Dibujar puntos clave en modo detallado
imagen_kp_rich = cv.drawKeypoints(imagen, kp, None, color=(0, 255, 0), 
                                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar resultados
cv.imshow('Puntos clave - Modo simple', imagen_kp_simple)
cv.imshow('Puntos clave - Modo detallado', imagen_kp_rich)
cv.waitKey(0)
cv.destroyAllWindows()

#
# Comparación entre descriptores de dos imágenes


# Crear una transformación de rotación
(h, w) = grises.shape[:2]
centro = (w // 2, h // 2)
matriz_rotacion = cv.getRotationMatrix2D(centro, 30, 1.0)  # Rotar 30 grados
grises_rotada = cv.warpAffine(grises, matriz_rotacion, (w, h))

# Detectar puntos clave y descriptores en la imagen rotada
kp2, des2 = sift.detectAndCompute(grises_rotada, None)

print(f"Puntos clave en imagen original: {len(kp)}")
print(f"Puntos clave en imagen rotada: {len(kp2)}")

# Configurar FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Número de verificaciones

flann = cv.FlannBasedMatcher(index_params, search_params)

# Encontrar coincidencias
matches = flann.knnMatch(des, des2, k=2)

# Aplicar test de ratio de Lowe
buenas_coincidencias = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        buenas_coincidencias.append(m)

print(f"Coincidencias buenas encontradas: {len(buenas_coincidencias)}")

# Dibujar coincidencias
imagen_coincidencias = cv.drawMatches(imagen, kp, 
                                    cv.cvtColor(grises_rotada, cv.COLOR_GRAY2BGR), kp2, 
                                    buenas_coincidencias, None, 
                                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Coincidencias FLANN', imagen_coincidencias)
cv.waitKey(0)
cv.destroyAllWindows()
