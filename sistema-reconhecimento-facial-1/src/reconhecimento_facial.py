import cv2
import sqlite3
import os
from deepface import DeepFace
from tkinter import messagebox

class ReconhecimentoFacial:
    """Reconhecimento facial usando ArcFace e validação de usuário."""

    def capturar_imagem(self):
        """Captura uma imagem da webcam."""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def tratar_imagem(self, imagem):
        """Redimensiona e equaliza histograma da imagem facial."""
        if imagem is None:
            print("Erro: imagem recebida é None!")
            raise ValueError("Imagem inválida para tratamento.")
        print("Tratando imagem: redimensionamento e equalização de histograma")
        imagem = cv2.resize(imagem, (224, 224))
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        imagem_eq = cv2.equalizeHist(imagem_cinza)
        imagem_blur = cv2.GaussianBlur(imagem_eq, (3, 3), 0)  # Novo: leve desfoque
        imagem_final = cv2.cvtColor(imagem_blur, cv2.COLOR_GRAY2BGR)
        return imagem_final

    def reconhecer_face(self, imagem, cpf=None):
        """
        Detecta e reconhece a face na imagem usando ArcFace.
        Se CPF for informado, compara apenas com as imagens desse usuário.
        Se não, compara com todos os usuários e retorna o CPF reconhecido.
        """
        print("Usando ArcFace para reconhecimento facial!")
        try:
            imagem_tratada = self.tratar_imagem(imagem)
        except Exception as e:
            print(f"Erro ao tratar imagem: {e}")
            messagebox.showerror("Erro", f"Falha ao tratar imagem: {e}")
            return None

        if cpf:
            usuario = self.buscar_usuario_por_cpf(cpf)
            if not usuario:
                print("Usuário não encontrado.")
                return None
            imagens_cadastradas = usuario[4].split(";")
            return self._verificar_imagens(imagem_tratada, imagens_cadastradas, cpf)
        else:
            conn = sqlite3.connect('cadastro.db')
            cursor = conn.cursor()
            cursor.execute('SELECT cpf, imagem_facial FROM usuarios')
            usuarios = cursor.fetchall()
            conn.close()
            for cpf_db, imagens_str in usuarios:
                imagens_cadastradas = imagens_str.split(";")
                resultado_cpf = self._verificar_imagens(imagem_tratada, imagens_cadastradas, cpf_db)
                if resultado_cpf:
                    return resultado_cpf
            print("Nenhum usuário reconhecido.")
            return None

    def _verificar_imagens(self, imagem_tratada, imagens_cadastradas, cpf):
        """Verifica se alguma das imagens cadastradas autentica o usuário."""
        for imagem_path in imagens_cadastradas:
            if not os.path.exists(imagem_path):
                continue
            try:
                resultado = DeepFace.verify(imagem_tratada, imagem_path, model_name='ArcFace')
                # Limiar mais rigoroso
                if resultado["verified"] and resultado["distance"] < 0.4:
                    acuracia = max(0, int((1 - resultado["distance"]) * 100))
                    print(f"Usuário autenticado! CPF: {cpf} | Acurácia: {acuracia}%")
                    return cpf
            except Exception as e:
                print(f"Erro na verificação: {e}")
                continue
        return None

    def buscar_usuario_por_cpf(self, cpf):
        """Busca usuário pelo CPF no banco de dados."""
        conn = sqlite3.connect('cadastro.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM usuarios WHERE cpf = ?', (cpf,))
        usuario = cursor.fetchone()
        conn.close()
        return usuario