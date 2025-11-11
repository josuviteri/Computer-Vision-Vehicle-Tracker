"""La segunda parte de la práctica introduce al alumno en la captura y almacenamiento de video
en tiempo real. Para ello deberá inicializar la cámara web a través de cv.VideoCapture(0),
mostrar en pantalla cada fotograma en tiempo real con cv.imshow() y configurar un archivo de
salida mediante cv.VideoWriter(), utilizando un códec FourCC como DIVX. La ejecución del
programa debe permitir la grabación del video hasta que el usuario presione la tecla q.
Como actividad complementaria, se propone que el estudiante experimente con diferentes
códecs de compresión (por ejemplo, XVID o MJPG) y modifique la resolución de captura. De esta
manera podrá observar cómo varía la calidad visual y el tamaño del archivo resultante,
reflexionando sobre las ventajas y limitaciones de cada configuración."""
import cv2 as cv

# -------------------------------
# 1. Inicializar la cámara
# -------------------------------
cap = cv.VideoCapture(0)  # 0 = webcam por defecto

if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit()

# Configurar resolución (ejemplo: 640x480)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# -------------------------------
# 2. Configurar el archivo de salida
# -------------------------------
# Elegir códec (puedes cambiar 'DIVX' por 'XVID' o 'MJPG')
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output_divx.avi', fourcc, 20.0, (640, 480))

# -------------------------------
# 3. Bucle de captura en tiempo real
# -------------------------------
print("Grabando... presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar fotograma.")
        break

    # Mostrar en pantalla
    cv.imshow('Video en tiempo real', frame)

    # Guardar en archivo
    out.write(frame)

    # Salir con tecla 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 4. Liberar recursos
# -------------------------------
cap.release()
out.release()
cv.destroyAllWindows()
