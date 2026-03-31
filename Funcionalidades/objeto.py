from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

CLASSES_PERMITIDAS = [56, 62, 63]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    resultados = model(frame, verbose=False)

    anomalia_detectada = False
    objetos_estranhos = []

    # Analisa tudo o que o YOLO encontrou na tela
    for resultado in resultados:
        for box in resultado.boxes:
            classe_id = int(box.cls[0])  # Pega o ID da classe detectada
            confianca = float(box.conf[0])  # Pega a certeza do modelo (0 a 1)

            # Só considera se o modelo tiver mais de 50% de certeza para evitar falso positivo
            if confianca > 0.5:
                if classe_id not in CLASSES_PERMITIDAS:
                    anomalia_detectada = True
                    nome_da_classe = model.names[classe_id]
                    objetos_estranhos.append(nome_da_classe)

    # Lógica do Alerta
    if anomalia_detectada:
        texto_alerta = f"ANOMALIA! Detectado: {', '.join(objetos_estranhos)}"
        print(texto_alerta)
        # Pinta a tela de vermelho e escreve o alerta
        cv2.putText(
            frame, texto_alerta, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    else:
        cv2.putText(
            frame,
            "Ambiente Padrao Seguro",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Monitoramento de Ambiente", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
