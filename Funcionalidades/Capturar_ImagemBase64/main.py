import cv2
import base64


def capturar_e_salvar_imagem():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Câmera acessada. Capturando imagem...")

    sucesso, frame = camera.read()

    if sucesso:
        nome_arquivo_imagem = "minha_foto.jpg"
        cv2.imwrite(nome_arquivo_imagem, frame)
        print(f"Imagem salva com sucesso como '{nome_arquivo_imagem}'")

        # Converter a imagem para base64
        _, buffer = cv2.imencode(".jpg", frame)
        imagem_base64 = base64.b64encode(buffer).decode("utf-8")

        nome_arquivo_base64 = "minha_foto_base64.txt"
        with open(nome_arquivo_base64, "w") as arquivo_texto:
            arquivo_texto.write(imagem_base64)
        print(f"String Base64 salva com sucesso como '{nome_arquivo_base64}'")

    else:
        print("Erro: Não foi possível capturar a imagem.")

    # Libera a câmera para que outros programas possam usá-la
    camera.release()


if __name__ == "__main__":
    capturar_e_salvar_imagem()
