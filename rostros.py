import os
import sys
import urllib.request
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DE_IMAGEN = os.path.join(BASE, sys.argv[1] if len (sys.argv) > 1 else "THE DONATO.png")
URL_YUNET = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
RUTA_YUNET = os.path.join(BASE, "face_detection_yunet_2023mar.onnx")


def detectar_rostros(gray):
    """Devuelve la lista de (x, y, w, h)"""
    if hasattr(cv2, "CascadeClassifier"):
        cascada = os.path.join(cv2.data.harcascades, "haarcascade_frontalface_default.xml")
        clasificador = cv2.CascadeClassifier(cascada)
        if not clasificador.empty():
            return list(clasificador.detectMultiScale(gray, 1.3, 5))
        print("Aviso: no se encontro la cascada de Haar, se usara YuNet.")

    if not os.path.exists(RUTA_YUNET):
        print("Descargando el modelo YuNet (solo si es descargado por primera vez)...")
        urllib.request.urlretrieve(URL_YUNET, RUTA_YUNET)

    detector = cv2.FaceDetectorYN.create(RUTA_YUNET, "", (gray.shape[1], gray.shape[0]),
                                     0.6, 0.3, 5000)
    faces = detector.detect(gray)[1]
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


img = cv2.imread(RUTA_DE_IMAGEN)

if img is None:
    print(f"Error: No se pudo cargar la imagen '{RUTA_DE_IMAGEN}'.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = detectar_rostros(gray)

for (x, y, w, h) in rostros:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

print(f"Rostros detectados: {len(rostros)}")

cv2.imshow("Deteccion de rostros", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
