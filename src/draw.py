import cv2


def draw_track(frame, track, color=(0, 255, 0)):
    # Dibuja bbox, etiqueta 'ID n | v px/f' y la trayectoria de puntos del track
    pts = list(track.trail)
    n = len(pts)
    for i, (px, py) in enumerate(pts):
        k = (i + 1) / n  # los puntos antiguos se ven menos
        c = tuple(int(v * (0.3 + 0.7 * k)) for v in color)
        cv2.circle(frame, (px, py), 2 + int(2 * k), c, -1)

    x, y, w, h = track.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"ID {track.id} | {track.speed:.1f} px/f", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
