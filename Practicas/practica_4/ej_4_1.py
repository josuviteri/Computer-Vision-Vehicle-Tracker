import cv2 as cv



imagen = cv.imread("media/twotone-circles.tif")


if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit()

# --- 2. Convertir a escala de grises ---
gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

# --- 3. Aplicar umbral binario ---
# cv.threshold devuelve: (valor_usado, imagen_umbralizada)
_, binaria = cv.threshold(gris, 127, 255, cv.THRESH_BINARY)
# Asegurar tipo correcto (threshold ya devuelve uint8, pero por seguridad):
binaria = binaria.astype('uint8')

# --- 4. Detectar contornos ---
# Puedes probar diferentes combinaciones:
# cv.RETR_TREE / cv.RETR_EXTERNAL
# cv.CHAIN_APPROX_SIMPLE / cv.CHAIN_APPROX_NONE
metodo_retr = cv.RETR_TREE
modo_aprox = cv.CHAIN_APPROX_SIMPLE

contornos, jerarquia = cv.findContours(binaria, metodo_retr, modo_aprox)

# --- 5. Dibujar contornos sobre la imagen original ---
imagen_contornos = imagen.copy()
cv.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

# --- 6. Calcular propiedades geométricas ---
for i, c in enumerate(contornos):
    M = cv.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    area = cv.contourArea(c)
    perimetro = cv.arcLength(c, True)

    print(f"Contorno {i}:")
    print(f" - Centroide: ({cx}, {cy})")
    print(f" - Área: {area:.2f}")
    print(f" - Perímetro: {perimetro:.2f}")

    # Dibujar centroide
    cv.circle(imagen_contornos, (cx, cy), 4, (0, 0, 255), -1)

#Resultados
cv.imshow("Original", imagen)
cv.imshow("Escala de grises", gris)
cv.imshow("Binaria", binaria)
cv.imshow("Contornos detectados", imagen_contornos)

cv.waitKey(0)
cv.destroyAllWindows()
