import cv2
import os
import sqlite3
from tkinter import messagebox
from deepface import DeepFace
import numpy as np

def salvar_embeddings(imagens, cpf):
    """Salva um embedding para cada imagem facial do usuário."""
    pasta = os.path.join("faces", cpf)
    os.makedirs(pasta, exist_ok=True)
    for i, img in enumerate(imagens):
        embedding = DeepFace.represent(img, model_name="ArcFace")[0]["embedding"]
        np.save(os.path.join(pasta, f"embedding_{i+1}.npy"), embedding)

class Cadastro:
    """Classe para cadastro de usuários com imagens faciais."""

    def __init__(self, nome, data_nascimento, cpf, imagens_faciais):
        self.nome = nome
        self.data_nascimento = data_nascimento
        self.cpf = cpf
        self.imagens_faciais = imagens_faciais  # lista de imagens

    def cadastrar_usuario(self):
        """Salva múltiplas imagens faciais e dados do usuário no banco."""
        pasta_cpf = os.path.join("faces", self.cpf)
        os.makedirs(pasta_cpf, exist_ok=True)
        imagem_paths = []
        for i, imagem in enumerate(self.imagens_faciais):
            imagem_path = os.path.join(pasta_cpf, f"{self.cpf}_{i+1}.jpg")
            # Garante que a imagem está no tamanho correto
            imagem = cv2.resize(imagem, (224, 224))
            cv2.imwrite(imagem_path, imagem)
            imagem_paths.append(imagem_path)
        imagens_str = ";".join(imagem_paths)

        conn = sqlite3.connect("cadastro.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                data_nascimento TEXT NOT NULL,
                cpf TEXT NOT NULL UNIQUE,
                imagem_facial TEXT NOT NULL
            )
        """)
        cursor.execute("SELECT * FROM usuarios WHERE cpf = ?", (self.cpf,))
        if cursor.fetchone():
            print("CPF já cadastrado!")
            conn.close()
            return False
        cursor.execute("""
            INSERT INTO usuarios (nome, data_nascimento, cpf, imagem_facial)
            VALUES (?, ?, ?, ?)
        """, (self.nome, self.data_nascimento, self.cpf, imagens_str))
        conn.commit()
        conn.close()
        # Salva embeddings de todas as imagens capturadas
        if self.imagens_faciais:
            salvar_embeddings(self.imagens_faciais, self.cpf)
        return True

    def validar_cpf(self):
        """Valida se o CPF possui 11 dígitos numéricos."""
        return self.cpf.isdigit() and len(self.cpf) == 11

    def capturar_imagens_guiado(self):
        """Captura 4 imagens faciais guiadas por instruções usando a webcam."""
        instrucoes = [
            "Olhe para a câmera e mantenha o rosto centralizado.",
            "Vire levemente o rosto para a ESQUERDA.",
            "Vire levemente o rosto para a DIREITA.",
            "Aproxime o rosto da câmera."
        ]
        fotos_capturadas = []
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for passo, instrucao in enumerate(instrucoes):
            messagebox.showinfo("Instrução", instrucao)
            while True:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Erro", "Erro ao acessar a câmera.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    break
                cv2.imshow("Cadastro Facial", frame)
                key = cv2.waitKey(1)
                if key == ord(' '):  # Espaço para capturar
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        rosto = frame[y:y+h, x:x+w]
                        imagem_tratada = cv2.resize(rosto, (224, 224))
                        fotos_capturadas.append(imagem_tratada)
                        messagebox.showinfo("Foto tirada", f"Foto {passo+1} capturada com sucesso!")
                        break
                    else:
                        messagebox.showwarning("Atenção", "Nenhum rosto detectado. Tente novamente.")
            cv2.destroyAllWindows()
        cap.release()
        return fotos_capturadas
