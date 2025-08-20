from deepface import DeepFace
import cv2
import os
import numpy as np
from utils.database import get_user_by_cpf, update_user

class Login:
    """
    Classe responsável por autenticação facial no sistema.
    Utiliza a biblioteca DeepFace para verificar similaridade entre o rosto
    capturado e os rostos salvos no banco de dados (e em disco).
    """

    def __init__(self):
        # Tamanho padrão usado para redimensionar imagens antes da verificação
        self.tamanho_imagem = (224, 224)
        # Pasta onde ficam armazenadas as imagens faciais dos usuários
        self.pasta_faces = "faces"

    # -------------------- Funções principais de autenticação --------------------

    def autenticar_facial(self):
        """
        Captura a imagem pela webcam e autentica o usuário com base no rosto.
        Mostra no terminal uma mensagem de sucesso ou falha.
        Retorna:
            bool: True se a autenticação for bem-sucedida, False caso contrário.
        """
        print("Capturando imagem para autenticação...")
        frame = self._capturar_imagem()
        if frame is None:
            print("Não foi possível capturar a imagem.")
            return False

        cpf_usuario = self._autenticar_por_imagem(frame)
        if cpf_usuario:
            print(f"Usuário autenticado! CPF: {cpf_usuario}")
            return True
        else:
            print("Falha na autenticação facial.")
            return False

    def autenticar_sem_cpf(self):
        """
        Captura imagem e retorna apenas o CPF do usuário autenticado (sem mensagens).
        Ideal para uso interno no sistema.
        Retorna:
            str | None: CPF do usuário se reconhecido, senão None.
        """
        frame = self._capturar_imagem()
        if frame is None:
            return None
        return self._autenticar_por_imagem(frame)

    def autenticar_com_acuracia(self):
        """
        Captura imagem e autentica retornando CPF + acurácia (confiança) da comparação.
        Útil para auditoria ou ajustes de limiar de aceitação.
        Retorna:
            tuple | None: (CPF, acurácia) se reconhecido, senão None.
        """
        frame = self._capturar_imagem()
        if frame is None:
            return None
        return self._autenticar_por_imagem(frame, retornar_acuracia=True)

    # -------------------- Função de autenticação central --------------------

    def _autenticar_por_imagem(self, imagem, retornar_acuracia=False):
        """
        Compara a imagem capturada com todas as imagens cadastradas no sistema.
        Percorre todos os usuários e suas fotos para encontrar correspondência.

        Parâmetros:
            imagem (np.ndarray): Imagem capturada (frame da webcam).
            retornar_acuracia (bool): Se True, retorna também a acurácia.

        Retorna:
            str | tuple | None:
                - CPF do usuário reconhecido (str)
                - (CPF, acurácia) se `retornar_acuracia=True`
                - None se não encontrar correspondência.
        """
        imagem = cv2.resize(imagem, self.tamanho_imagem)

        for usuario in get_user_by_cpf():
            cpf = usuario['cpf']
            pasta_usuario = os.path.join(self.pasta_faces, cpf)

            if not os.path.exists(pasta_usuario):
                continue

            for arquivo in os.listdir(pasta_usuario):
                caminho_img_cadastrada = os.path.join(pasta_usuario, arquivo)

                try:
                    resultado = DeepFace.verify(
                        img1_path=imagem,
                        img2_path=caminho_img_cadastrada,
                        model_name="ArcFace",
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                except Exception:
                    continue

                if resultado["verified"]:
                    if retornar_acuracia:
                        return cpf, resultado["distance"]
                    return cpf

        return None

    # -------------------- Funções utilitárias --------------------

    def _capturar_imagem(self):
        """
        Abre a webcam e captura um único frame.
        Retorna:
            np.ndarray | None: Frame capturado ou None se não conseguir.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    def salvar_imagens_cadastradas(self, cpf, imagens):
        """
        Salva as imagens faciais de um usuário no disco e atualiza o banco de dados.

        Parâmetros:
            cpf (str): CPF do usuário.
            imagens (list[np.ndarray]): Lista de imagens (já processadas).
        """
        pasta_usuario = os.path.join(self.pasta_faces, cpf)
        os.makedirs(pasta_usuario, exist_ok=True)

        for i, img in enumerate(imagens):
            caminho = os.path.join(pasta_usuario, f"{cpf}_{i+1}.jpg")
            cv2.imwrite(caminho, img)

        update_user(cpf, fotos=len(imagens))
