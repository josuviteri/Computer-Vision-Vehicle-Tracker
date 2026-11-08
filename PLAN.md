# Plan de mejora — Vehicle Tracker (CV clásico)

> Restricción: **sin ML/DL**. Solo técnicas clásicas de OpenCV (sustracción de fondo, morfología, KCF/CSRT, filtro de Kalman, homografía…).
---

## Diagnóstico del script actual (`original_experiments/developed_approaches/kfc_developed.py`)

| Problema | Consecuencia |
|---|---|
| Si `tracker.update()` falla en un solo frame, el track se **elimina inmediatamente** | El vehículo se "pierde" y al redetectarlo recibe **un ID nuevo** (lo que señala el profesor) |
| `car_total_count` se incrementa por cada tracker creado | El conteo total se infla con cada re-adquisición |
| `backSub.apply()` solo se ejecuta cada 10 frames | MOG2 aprende mal el fondo → máscaras ruidosas → detecciones inestables |
| Las detecciones solo se descartan "si están a < 50 px de un tracker" (posición actual, no predicha) | No hay asociación real detección↔track; no se aprovecha el historial de movimiento |
| KCF nunca se re-ancla a la detección | El bbox deriva poco a poco hasta quedarse pegado al fondo o perder el coche |
| ROI, resolución (1280×720) y umbrales de ángulo fijos en código | Solo funciona para el cruce de Buenos Aires |
| Dirección = recta entre primeros y últimos puntos, en píxeles | Sensible al ruido y a la perspectiva; los ángulos no son comparables entre cámaras |

---

## Estructura propuesta

Pasar del script único a un pequeño paquete (sin sobre-ingeniería):

```
src/
  main.py          # CLI: --url <webcamtaxi> | --video <file.mp4>, --roi <json>
  stream.py        # Fase 0: resolución de URL + lector en hilo
  roi.py           # Fase 1: selección interactiva de ROI (y calibración en fase 3)
  detector.py      # Sustracción de fondo + morfología → lista de bboxes
  tracker.py       # Fase 2: clase Track + gestor de tracks
  analytics.py     # Fase 3: homografía, velocidad, dirección, trayectorias
  draw.py          # Overlay / HUD
configs/           # ROI y calibración guardados por cámara (JSON)
clips/             # Grabaciones locales para pruebas reproducibles (en .gitignore)
requirements.txt   # opencv-contrib-python, numpy, scipy, yt-dlp, requests
```

> Nota: `cv2.TrackerKCF_create` requiere **`opencv-contrib-python`** en versiones recientes de OpenCV.

---

## Fase 0 — Procesamiento de vídeo en streaming

**Objetivo:** dada una URL de una página de https://www.webcamtaxi.com/, procesar el directo en tiempo real.

### Tareas
- [ ] **Resolver la URL del stream.** Las páginas de webcamtaxi normalmente incrustan un **directo de YouTube** (iframe) o, a veces, un **HLS `.m3u8`**.
  1. Descargar el HTML de la página (`requests`) y extraer con regex el `src` del iframe de YouTube o la URL `.m3u8`.
  2. Si es YouTube → obtener la URL HLS directa con `yt-dlp -g -f "best[height<=720]"` (vía su API de Python).
  3. Si es `.m3u8` → usarla tal cual.
  4. Fallback: permitir pasar directamente una URL de YouTube/m3u8 o un `.mp4` local (`--video`).
- [ ] **Lector en hilo (`StreamReader`).** `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` en un hilo que lee continuamente y guarda **solo el último frame** (cola de tamaño 1). Así, si el procesamiento va más lento que el stream, se descartan frames en vez de acumular retraso.
- [ ] **Reconexión.** Si `read()` falla N veces seguidas → reabrir la captura (las URLs HLS de YouTube caducan; volver a resolverlas con `yt-dlp`).
- [ ] **Timestamps reales.** Guardar `time.monotonic()` con cada frame. El FPS de los streams es variable y la fase 3 necesita Δt real para calcular velocidades.
- [ ] **Grabación de clips (`--record clips/x.mp4`).** Guardar unos minutos del directo para **probar siempre con el mismo material**. Es imprescindible para comparar "antes vs después" en la fase 2, porque un directo no se puede repetir.
- [ ] Redimensionar a un ancho de trabajo fijo (p. ej. 960 px) manteniendo la relación de aspecto, en lugar de forzar 1280×720.

### Criterio de aceptación
`python src/main.py --url https://www.webcamtaxi.com/...` muestra el directo con un retraso que no aumenta con el tiempo, y se recupera solo si se corta la conexión.

---

## Fase 1 — Selección manual del ROI sobre el primer frame

**Objetivo:** que el ROI deje de estar fijado en el código.

### Tareas
- [ ] Al arrancar, capturar el primer frame válido y mostrarlo en una ventana de selección.
- [ ] Selección de **polígono** con `cv2.setMouseCallback`:
  - clic izquierdo → añadir vértice (se dibuja en vivo)
  - `z` → deshacer el último punto · `r` → reiniciar · `Enter` → confirmar (mínimo 3 puntos) · `Esc` → salir
- [ ] Generar la máscara binaria a partir del polígono (`cv2.fillPoly`) con el tamaño real del frame.
- [ ] **Guardar el ROI** en `configs/<camara>.json` (nombre derivado de la URL). En siguientes ejecuciones, preguntar si se reutiliza o se dibuja uno nuevo (`--roi configs/x.json` para cargarlo directamente).
- [ ] Guardar las coordenadas **normalizadas (0–1)** para que el ROI siga siendo válido si cambia la resolución del stream.
- [ ] (Opcional) Línea de conteo: tras el ROI, marcar 2 puntos para una línea virtual de cruce (ver fase 2).

### Criterio de aceptación
Se puede usar el sistema sobre cualquier cámara nueva sin tocar el código.

---

## Fase 2 — Tracking más robusto

**Objetivo:** que un mismo vehículo **conserve su ID** de principio a fin y que **se cuente una sola vez**.

Idea central (sugerencia del profesor): *un vehículo no desaparece de repente* → combinar **varias fuentes**:
- **Detector** (sustracción de fondo): dice *dónde hay algo moviéndose*.
- **KCF**: dice *dónde está este coche en concreto* (apariencia), frame a frame.
- **Filtro de Kalman** (velocidad constante, `cv2.KalmanFilter`): dice *dónde debería estar* según su movimiento previo, y permite "rellenar" los huecos cuando KCF y el detector fallan.

### 2.1 Detector
- [ ] Llamar a `backSub.apply()` **en todos los frames** (es barato); usar la máscara solo cada N frames para detectar.
- [ ] Mantener el pipeline actual (sombras fuera, blur, umbral, apertura/cierre), y añadir un filtro de **relación de aspecto** y de **solidez** para descartar ruido.
- [ ] Hacer el área mínima/máxima **relativa al área del ROI** (o configurable) en lugar de 120–2000 px fijos.

### 2.2 Clase `Track`
Estado por vehículo:
```
id, bbox, kcf, kalman, history[(t, cx, cy)], hits, misses, state ∈ {TENTATIVE, CONFIRMED, LOST}
```

### 2.3 Ciclo por frame
1. **Predicción:** `kalman.predict()` para todos los tracks.
2. **KCF:** `kcf.update(frame)`.
   - Si funciona y el bbox es coherente con la predicción de Kalman (distancia < umbral) → `kalman.correct()` con el centro de KCF.
   - Si falla o hace un salto incoherente → **no se borra el track**: se marca un fallo (`misses += 1`) y se usa la posición predicha por Kalman ("coasting").
3. **Detección (cada N frames):** asociar detecciones a tracks.
   - Matriz de coste = `1 − IoU` (o distancia entre centros) usando la **posición predicha**, no la última.
   - Asignación con **húngaro** (`scipy.optimize.linear_sum_assignment`), con un umbral de coste máximo.
   - Detección emparejada → `kalman.correct()`, **re-inicializar KCF con el bbox detectado** (corrige la deriva), `misses = 0`, `hits += 1`.
   - Detección sin emparejar → primero intentar **re-identificar** un track `LOST` reciente cerca de su posición predicha (y, opcionalmente, con un histograma HSV parecido, `cv2.compareHist`). Si no hay ninguno → nuevo track `TENTATIVE`.
4. **Ciclo de vida:**
   - `TENTATIVE → CONFIRMED` tras `M` aciertos (p. ej. 3). **Solo se cuentan y se dibujan los confirmados** → se eliminan los falsos positivos de un solo frame.
   - `CONFIRMED → LOST` si `misses > K1`; se mantiene un tiempo en memoria para poder re-identificarlo.
   - `LOST → eliminado` si `misses > K2`, si el centro predicho sale del ROI o si el track lleva mucho tiempo **sin moverse** (KCF pegado al fondo).
5. **Supresión de duplicados:** si dos tracks se solapan mucho (IoU > 0.6) durante varios frames → fusionar y conservar el ID más antiguo.

### 2.4 Conteo
- [ ] Contar un vehículo cuando pasa a `CONFIRMED` (o, si se definió en la fase 1, cuando su trayectoria **cruza la línea de conteo**). Así el total deja de inflarse con las re-adquisiciones.

### 2.5 Evaluación (ligera)
- [ ] Sobre 1–2 clips grabados en la fase 0, comparar el script original con el nuevo:
  - nº de IDs generados vs. nº real de vehículos (contado a mano)
  - nº de cambios de ID observados
- [ ] Una tabla pequeña en el README es suficiente.

### Parámetros a exponer (no hardcodear)
`detect_every`, `min_hits (M)`, `max_misses_lost (K1)`, `max_misses_delete (K2)`, `max_assoc_cost`, `min_area`, `max_area`, tipo de tracker (`KCF` / `CSRT`).

### Criterio de aceptación
En los clips de prueba, la mayoría de los vehículos que cruzan el ROI conservan **un único ID** y el total contado se acerca al real.

---

## Fase 3 — Analítica: velocidad, dirección y trayectoria

### 3.1 Calibración (introducción manual de distancias)
- [ ] Nuevo paso tras el ROI: marcar **4 puntos sobre el suelo** que formen un rectángulo real (p. ej. los bordes de un carril o un paso de cebra) e introducir por consola su **ancho y largo en metros**. Si el ROI tiene 4 puntos, permitir reutilizarlo.
- [ ] Calcular la **homografía** `H = cv2.getPerspectiveTransform(pts_img, pts_metros)`.
- [ ] Guardar `H` y los puntos en el mismo `configs/<camara>.json`.
- [ ] Función `to_world(cx, cy)` → `cv2.perspectiveTransform` → coordenadas en metros sobre el plano del suelo.
- [ ] Usar como punto de referencia el **centro inferior del bbox** (el contacto con el suelo), no el centro, para reducir el error de perspectiva.

### 3.2 Velocidad
- [ ] Para cada track `CONFIRMED`: tomar las posiciones en metros y los timestamps (fase 0) de los últimos `W` puntos (≈ 0.5–1 s).
- [ ] `v = distancia recorrida / Δt` → km/h. Suavizar con una media móvil o usar directamente la velocidad que estima el filtro de Kalman (convertida a metros).
- [ ] No mostrar la velocidad hasta tener suficiente historial (evita valores absurdos al principio).
- [ ] Guardar la velocidad media y la máxima por vehículo.

### 3.3 Dirección (vector)
- [ ] Sustituir "primer punto → último punto" por un **vector de dirección** ajustado con los últimos `N` puntos: `cv2.fitLine` o mínimos cuadrados, y el sentido según el signo del desplazamiento neto.
- [ ] Calcularlo en **coordenadas del suelo** (tras la homografía) → ángulos comparables y sin la distorsión de la perspectiva.
- [ ] Dibujar una **flecha** (`cv2.arrowedLine`) desde el vehículo con su dirección.
- [ ] Generalizar la clasificación: en vez de los rangos de grados fijos del cruce de Buenos Aires, clasificar respecto a los ejes del rectángulo de calibración (→ "sentido A/B a lo largo de la vía", "cruzando"), o en 4/8 sectores (N, NE, E…) en la vista cenital.

### 3.4 Trayectoria
- [ ] Dibujar la trayectoria suavizada de cada track (coloreada por velocidad u ID).
- [ ] Capa acumulada con las trayectorias de todos los vehículos (mapa de flujo) que se pueda mostrar u ocultar con una tecla.
- [ ] (Opcional) Vista **cenital** en una ventana pequeña con `cv2.warpPerspective` + los puntos de los vehículos.
- [ ] Exportar a **CSV** al terminar: `id, t_inicio, t_fin, velocidad_media, velocidad_max, dirección, puntos…`.

### 3.5 HUD
- [ ] Total de vehículos, vehículos activos, conteo por sentido, velocidad media de la última ventana y FPS de procesamiento.

### Criterio de aceptación
Con una cámara calibrada, las velocidades mostradas son del orden esperado para la vía (p. ej. 20–60 km/h en ciudad) y las flechas siguen la dirección real de los vehículos.

---

## Orden de trabajo y riesgos

1. **Fase 0 → 1** primero: sin stream ni ROI no se puede probar nada. Grabar clips cuanto antes.
2. **Fase 2**: el núcleo. Iterar sobre los clips grabados, no sobre el directo.
3. **Fase 3**: depende de tener IDs estables (si no, las velocidades salen con ruido).

| Riesgo | Mitigación |
|---|---|
| webcamtaxi cambia el embed o la cámara está offline | Aceptar también URLs de YouTube/m3u8 directas y `.mp4` locales |
| Las URLs HLS de YouTube caducan | Volver a resolverlas con `yt-dlp` al reconectar |
| Baja resolución o bitrate y cambios de luz (noche, lluvia) | Elegir cámaras diurnas para las pruebas; documentarlo como limitación |
| `cv2.imshow` en WSL2 | Requiere WSLg (Windows 11); si no, ejecutar en Windows nativo |
| Calibración imprecisa → velocidades con error | Usar marcas viales de medida conocida; presentar la velocidad como "aproximada" |

## Decisiones abiertas
- [ ] Qué cámara(s) de webcamtaxi usar
- [ ] Contar por confirmación o por cruce de línea (propuesta: línea si la vía lo permite).
- [ ] KCF frente a CSRT: CSRT es más preciso pero más lento; decidirlo midiendo FPS en la fase 2.
