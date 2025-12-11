***

# Detector y Rastreo de Vehículos - Visión por Computador

Este proyecto implementa un sistema de visión por computador capaz de detectar, rastrear y contar vehículos en un cruce de tráfico real utilizando técnicas de procesamiento de video tradicionales y eficientes.

## Descripción del Proyecto

El objetivo principal es desarrollar un sistema que pueda detectar y rastrear diversos tipos de vehículos (coches, autobuses, motocicletas) mediante una cámara fija orientada a un cruce. El sistema no solo realiza el seguimiento, sino que también:
* Cuenta cuántos vehículos circulan en cada dirección.
* Filtra detecciones por área mínima para evitar falsos positivos.
* Determina el sentido y el ángulo de desplazamiento.

Esta herramienta está diseñada para ser útil en la gestión de tráfico, permitiendo potencialmente el control adaptativo de semáforos para mejorar la fluidez y evitar accidentes.

## Origen de los Datos

* **Fuente:** Cámara de vigilancia pública.
* **Ubicación:** Un cruce en Buenos Aires, Argentina.
* **Condiciones:** Perspectiva fija con iluminación variable (clima/hora), tráfico continuo y movimientos de cámara ocasionales por viento.

## Metodología y Tecnología

A diferencia de las soluciones modernas basadas en Deep Learning (como YOLO), este proyecto opta por un enfoque clásico debido a restricciones de hardware (falta de GPU dedicada) y requisitos académicos.


### Algoritmo Seleccionado: KCF
Se evaluaron diversos algoritmos de seguimiento incluidos en OpenCV (`BOOSTING`, `MIL`, `TLD`, `MEDIANFLOW`, `GOTURN`, `MOSSE`, `CSRT`).
* **Elección Final:** **KCF (Kernelized Correlation Filter)**.
* **Motivo:** Ofrece el mejor equilibrio entre rendimiento y precisión en tiempo real sin necesidad de redes neuronales.
* **Ventajas:** Suaviza el seguimiento y evita la duplicación de detecciones en comparación con métodos puramente basados en diferencia de píxeles.

## Lógica del Sistema

El flujo de trabajo del algoritmo es el siguiente:

1.  **Inicialización:** Carga del video y definición de una resolución predefinida.
2.  **Región de Interés (ROI):** Se aplica un ROI para centrar el procesamiento en la zona activa del cruce.
3.  **Detección y Filtrado:**
    * Se utilizan variables globales para establecer distancias y áreas mínimas/máximas entre vehículos.
    * Se filtran detecciones excesivamente cercanas o con áreas fuera de los límites.
4.  **Seguimiento (Tracking):**
    * El tracker KCF se actualiza cada **5 frames** (en lugar de cada uno) para mitigar pérdidas cuando los vehículos aceleran o maniobran bruscamente.
    * Se asigna un ID único a cada vehículo.
5.  **Cálculo de Trayectoria:**
    * Se guardan las posiciones de los últimos 40 frames.
    * Se calcula una línea entre el primer y último punto para obtener el ángulo (valor entre 0 y 356) y el sentido del desplazamiento.

## Visualización y Resultados

El sistema muestra el video procesado en tiempo real con la siguiente información superpuesta:

* **Bounding Box:** Dibuja un cubo alrededor del vehículo con su ID y ángulo de trayectoria.
* **Código de Colores:** Para identificar la dirección visualmente:
    * **Azul Claro:** Movimiento en Eje X.
    * **Azul Oscuro:** Movimiento en Eje Y.
    * **Azul Neutro:** Transición entre ejes.
* **Estadísticas en Pantalla:**
    * Recuento total de vehículos.
    * Vehículos presentes en el instante.
    * Desglose de movimiento por ejes (X e Y) y direcciones (Izquierda/Derecha).

### Limitaciones Conocidas
* **Oclusiones:** Aglomeraciones de vehículos en puntos muy cercanos pueden causar fallos.
* **Pérdida de Trackers:** Puede ocurrir en maniobras muy bruscas, aunque se mitigó ajustando la tasa de actualización del tracker.
* **Iluminación:** Sensible a variaciones fuertes de luz.

## Referencias

* Bradski, G. and Kaehler, A. *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.
* Documentación oficial de OpenCV.
* S. S. Sahar, "Object Tracking Using KCF Algorithm." *International Journal of Computer Applications*, 2020.
* Material docente de Procesamiento de Vídeo, Universidad de Deusto.

---
*Josu Viteri | 2025/12*