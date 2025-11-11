
import cv2 as cv
import numpy as np

# --- 1. Cargar la imagen ---
img = cv.imread('media/woman-and-baby.tif')

if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit()

# --- 2. Definir el rectángulo inicial ---
# Formato: (x, y, ancho, alto)
# Ajusta estos valores según tu imagen para incluir el objeto completo
rect = (50, 50, 400, 500)

# --- 3. Crear la máscara y los modelos ---
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)  # modelo de fondo
fgdModel = np.zeros((1, 65), np.float64)  # modelo de primer plano

# --- 4. Aplicar GrabCut ---
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# --- 5. Crear máscara binaria: 1 = primer plano, 0 = fondo ---
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# --- 6. Multiplicar por la imagen original ---
resultado = img * mask2[:, :, np.newaxis]

# --- 7. Dibujar el rectángulo sobre una copia para visualizar ---
img_rect = img.copy()
cv.rectangle(img_rect, (rect[0], rect[1]), 
            (rect[0] + rect[2], rect[1] + rect[3]), 
            (0, 255, 0), 2)

# --- 8. Mostrar los resultados ---
cv.imshow('Imagen original con rectángulo', img_rect)
cv.imshow('Máscara generada', mask2 * 255)
cv.imshow('Resultado GrabCut', resultado)

cv.waitKey(0)
cv.destroyAllWindows()
