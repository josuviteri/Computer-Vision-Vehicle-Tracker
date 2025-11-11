import cv2 as cv

# -------------------------------
# Función auxiliar para crear ROI
# -------------------------------
def createROI(img, x, y, w, h):
    """
    Recorta una región de interés (ROI) de una imagen.
    :param img: imagen original
    :param x: coordenada X (columna) inicial
    :param y: coordenada Y (fila) inicial
    :param w: ancho de la ROI
    :param h: alto de la ROI
    :return: subimagen (ROI)
    """
    return img[y:y+h, x:x+w]

# -------------------------------
# 1. Cargar la imagen
# -------------------------------
img = cv.imread("../../media/lenna-gray.tif")  # asegúrate de tener lenna.tif en la misma carpeta


# Mostrar imagen original
cv.imshow("Imagen original", img)

# -------------------------------
# 2. Extraer varias ROIs
# -------------------------------

# ROI centrada de 400x400
h, w, _ = img.shape
x_center = w//2 - 200
y_center = h//2 - 200
roi_centro = createROI(img, x_center, y_center, 400, 400)
cv.imshow("ROI centro", roi_centro)
cv.imwrite("roi_centro.png", roi_centro)

# ROI esquina superior izquierda
roi_sup_izq = createROI(img, 0, 0, 400, 400)
cv.imshow("ROI superior izquierda", roi_sup_izq)
cv.imwrite("roi_sup_izq.png", roi_sup_izq)

# ROI esquina inferior derecha
roi_inf_der = createROI(img, w-400, h-400, 400, 400)
cv.imshow("ROI inferior derecha", roi_inf_der)
cv.imwrite("roi_inf_der.png", roi_inf_der)

# ROI desplazada (ejemplo libre: un poco hacia la derecha y abajo)
roi_custom = createROI(img, 100, 150, 400, 400)
cv.imshow("ROI personalizada", roi_custom)
cv.imwrite("roi_custom.png", roi_custom)

# -------------------------------
# 3. Esperar y cerrar ventanas
# -------------------------------
cv.waitKey(0)
cv.destroyAllWindows()
