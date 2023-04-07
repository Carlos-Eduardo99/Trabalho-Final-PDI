import cv2
import numpy as np
import os

def detect_plate(img):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplica o algoritmo de detecção de bordas Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    # Encontra os contornos na imagem
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Verifica se o contorno tem mais de 4 pontos (evita detecção de ruído)
        if len(cnt) > 4:
            # Cria uma máscara para o contorno encontrado
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            # Calcula a média dos pixels dentro do contorno
            mean = cv2.mean(img, mask=mask)[0]
            # Verifica se a média dos pixels é próxima de branco (indicando uma placa de carro)
            if mean > 150:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = img[y:y+h, x:x+w]
                return roi
    return None

# Le todas as imagens contidas na pasta "/home/carlos/Área de Trabalho/Trabalho 3/imagens_placas"
images_path = "/home/carlos/Área de Trabalho/Trabalho 3/imagens_placas"
images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
for i, image in enumerate(images):
    img = cv2.imread(os.path.join(images_path, image))
    roi = detect_plate(img)
    if roi is not None:
        # Salva a subimagem extraída da imagem original no caminho "/home/carlos/Área de Trabalho/Trabalho 3/Subimagens"
        cv2.imwrite("/home/carlos/Área de Trabalho/Trabalho 3/Subimagens/placa_{}.png".format(i), roi)
