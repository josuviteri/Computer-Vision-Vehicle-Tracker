import cv2 as cv
import numpy as np

# --- 1. Cargar la imagen ---
img = cv.imread('media/balls-with-reflections.tif')
if img is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# --- 2. Convertir a escala de grises ---
gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# --- 3. Aplicar umbral de Otsu ---
_, binaria = cv.threshold(gris, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# --- 4. Definir kernel para operaciones morfológicas ---
kernel = np.ones((3, 3), np.uint8)

# --- 5. Primer plano seguro mediante erosión ---
fg = cv.erode(binaria, kernel, iterations=2)

# --- 6. Fondo seguro mediante dilatación ---
bg = cv.dilate(binaria, kernel, iterations=3)

# --- 7. Región desconocida ---
unknown = cv.subtract(bg, fg)

# --- 8. Etiquetado de marcadores ---
# Cada objeto debe tener un número único
_, markers = cv.connectedComponents(fg)

# Incrementar todos los marcadores para que el fondo sea 1 en lugar de 0
markers = markers + 1

# La región desconocida se marca con 0
markers[unknown == 255] = 0

# --- 9. Aplicar Watershed ---
markers = cv.watershed(img, markers)

# --- 10. Visualizar resultados ---
img_watershed = img.copy()
# Las líneas fronterizas se marcan en rojo
img_watershed[markers == -1] = [0, 0, 255]

cv.imshow('Imagen original', img)
cv.imshow('Binaria Otsu', binaria)
cv.imshow('Primer plano seguro', fg)
cv.imshow('Fondo seguro', bg)
cv.imshow('Región desconocida', unknown)
cv.imshow('Watershed', img_watershed)

cv.waitKey(0)
cv.destroyAllWindows()
