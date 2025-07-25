import cv2
import os
import sqlite3
from tkinter import messagebox
from deepface import DeepFace
import numpy as np

def salvar_embedding(frame, cpf):
    embedding = DeepFace.represent(frame, model_name="ArcFace")[0]["embedding"]
    pasta = os.path.join("faces", cpf)
    os.makedirs(pasta, exist_ok=True)
    np.save(os.path.join(pasta, "embedding.npy"), embedding)

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
        # Após salvar as imagens, salve o embedding do primeiro rosto capturado:
        if self.imagens_faciais:
            salvar_embedding(self.imagens_faciais[0], self.cpf)
        return True

    def validar_cpf(self):
        """Valida se o CPF possui 11 dígitos numéricos."""
        return self.cpf.isdigit() and len(self.cpf) == 11

    def capturar_imagens_automatico(self, total_fotos=30):
        """Captura automaticamente várias imagens faciais usando a webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        fotos_capturadas = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0

        while count < total_fotos:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Erro", "Erro ao acessar a câmera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                rosto = frame[y:y+h, x:x+w]
                # Redimensiona a imagem para 224x224 antes de salvar
                imagem_tratada = cv2.resize(rosto, (224, 224))
                fotos_capturadas.append(imagem_tratada)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Capturando {count}/{total_fotos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "Rosto não detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Captura Automática de Imagem Facial", frame)
            cv2.waitKey(200 if len(faces) > 0 else 1)

        cap.release()
        cv2.destroyAllWindows()
        if len(fotos_capturadas) == total_fotos:
            messagebox.showinfo("Captura", "Todas as fotos foram capturadas com sucesso!")
        else:
            messagebox.showwarning("Captura", f"Foram capturadas apenas {len(fotos_capturadas)} fotos.")
        return fotos_capturadas
