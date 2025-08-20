import cv2
import sqlite3
import os
from deepface import DeepFace
from tkinter import messagebox

class ReconhecimentoFacial:
    """Reconhecimento facial usando ArcFace e validação de usuário."""

    def capturar_imagem(self):
        """Captura uma imagem da webcam."""
        cap = cv2.VideoCapture(0)              # Abre a webcam padrão
        ret, frame = cap.read()                # Captura um frame da webcam
        cap.release()                         # Libera a webcam
        return frame if ret else None          # Retorna frame se capturado, senão None

    def tratar_imagem(self, imagem):
        """Redimensiona e equaliza histograma da imagem facial."""
        if imagem is None:
            print("Erro: imagem recebida é None!")
            raise ValueError("Imagem inválida para tratamento.")  # Verifica validade da imagem

        print("Tratando imagem: redimensionamento e equalização de histograma")
        imagem = cv2.resize(imagem, (224, 224))                       # Redimensiona para 224x224 pixels
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)       # Converte para escala de cinza
        imagem_eq = cv2.equalizeHist(imagem_cinza)                    # Equaliza o histograma para melhorar contraste
        imagem_blur = cv2.GaussianBlur(imagem_eq, (3, 3), 0)          # Aplica leve desfoque gaussiano para reduzir ruídos
        imagem_final = cv2.cvtColor(imagem_blur, cv2.COLOR_GRAY2BGR)  # Converte de volta para 3 canais (BGR)
        return imagem_final                                            # Retorna a imagem tratada

    def reconhecer_face(self, imagem, cpf=None):
        """
        Detecta e reconhece a face na imagem usando ArcFace.
        Se CPF for informado, compara apenas com as imagens desse usuário.
        Se não, compara com todos os usuários e retorna o CPF reconhecido.
        """
        print("Usando ArcFace para reconhecimento facial!")
        try:
            imagem_tratada = self.tratar_imagem(imagem)                # Tenta tratar a imagem recebida
        except Exception as e:
            print(f"Erro ao tratar imagem: {e}")
            messagebox.showerror("Erro", f"Falha ao tratar imagem: {e}")  # Mostra erro na GUI se falhar
            return None

        if cpf:                                                        # Se CPF foi passado
            usuario = self.buscar_usuario_por_cpf(cpf)                 # Busca usuário no banco
            if not usuario:
                print("Usuário não encontrado.")
                return None
            imagens_cadastradas = usuario[4].split(";")                # Pega lista de imagens do usuário
            return self._verificar_imagens(imagem_tratada, imagens_cadastradas, cpf)  # Verifica autenticação
        else:                                                          # Se CPF não foi passado
            conn = sqlite3.connect('cadastro.db')                      # Abre conexão com banco
            cursor = conn.cursor()
            cursor.execute('SELECT cpf, imagem_facial FROM usuarios')  # Busca todos os usuários e imagens
            usuarios = cursor.fetchall()
            conn.close()
            for cpf_db, imagens_str in usuarios:                       # Para cada usuário
                imagens_cadastradas = imagens_str.split(";")           # Lista de imagens dele
                resultado_cpf = self._verificar_imagens(imagem_tratada, imagens_cadastradas, cpf_db)  # Tenta autenticar
                if resultado_cpf:                                      # Se autenticado, retorna CPF
                    return resultado_cpf
            print("Nenhum usuário reconhecido.")
            return None

    def _verificar_imagens(self, imagem_tratada, imagens_cadastradas, cpf):
        """Verifica se alguma das imagens cadastradas autentica o usuário."""
        for imagem_path in imagens_cadastradas:                       # Para cada imagem cadastrada
            if not os.path.exists(imagem_path):                        # Se não existir arquivo, pula
                continue
            try:
                resultado = DeepFace.verify(imagem_tratada, imagem_path, model_name='ArcFace')  # Compara faces
                if resultado["verified"] and resultado["distance"] < 0.4:                     # Limite rigoroso
                    acuracia = max(0, int((1 - resultado["distance"]) * 100))                 # Calcula acurácia %
                    print(f"Usuário autenticado! CPF: {cpf} | Acurácia: {acuracia}%")
                    return cpf                                                                  # Retorna CPF autenticado
            except Exception as e:
                print(f"Erro na verificação: {e}")                                             # Imprime erro e segue
                continue
        return None                                                                               # Se não autenticar nenhuma, retorna None

    def buscar_usuario_por_cpf(self, cpf):
        """Busca usuário pelo CPF no banco de dados."""
        conn = sqlite3.connect('cadastro.db')                # Abre conexão com banco SQLite
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM usuarios WHERE cpf = ?', (cpf,))  # Consulta por CPF
        usuario = cursor.fetchone()                           # Busca o registro do usuário
        conn.close()                                          # Fecha conexão
        return usuario                                        # Retorna os dados ou None se não encontrado
