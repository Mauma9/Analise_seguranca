from ultralytics import YOLO
import cv2

# Carrega o modelo padrão do YOLOv8
model = YOLO("yolov8n.pt")

# Inicia a webcam (ou você pode colocar o caminho de um vídeo de teste em .mp4 aqui)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    # Roda o modelo no frame atual
    resultados = model(frame, verbose=False)

    for resultado in resultados:
        for box in resultado.boxes:
            classe_id = int(box.cls[0])
            confianca = float(box.conf[0])

            # A classe 0 no dataset COCO é sempre "person" (pessoa)
            if classe_id == 0 and confianca > 0.5:

                # Pegando as coordenadas da caixa: x1, y1 (topo esquerdo) e x2, y2 (base direita)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculando a largura e a altura da caixa da pessoa
                largura = x2 - x1
                altura = y2 - y1

                # HEURÍSTICA DE QUEDA:
                # Se a largura for consideravelmente maior que a altura, assumimos queda.
                # O multiplicador 1.2 é uma margem de segurança. Ajuste conforme o ângulo da sua câmera!
                if largura > altura * 1.2:
                    estado = "ALERTA: QUEDA DETECTADA!"
                    cor = (0, 0, 255)  # Vermelho
                else:
                    estado = "Pessoa Normal"
                    cor = (0, 255, 0)  # Verde

                # Desenhando a caixa e o texto na tela
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(
                    frame, estado, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2
                )

    # Mostra a imagem na tela
    cv2.imshow("Detector de Quedas", frame)

    # Aperte 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
