# ---------------------------
# IMPORTAÇÕES
# ---------------------------
import cv2              # OpenCV - usado para capturar e processar imagens/vídeo
import os               # Para manipular pastas e caminhos de arquivos
import sqlite3          # Banco de dados SQLite (armazenamento local)
from tkinter import messagebox  # Pop-ups para mostrar mensagens ao usuário
from deepface import DeepFace   # Biblioteca para reconhecimento facial e geração de embeddings
import numpy as np      # Biblioteca para manipulação de vetores/matrizes (usada para salvar embeddings)


# ---------------------------
# FUNÇÃO PARA SALVAR EMBEDDINGS
# ---------------------------
def salvar_embeddings(imagens, cpf):
    """
    Gera e salva um arquivo de embedding (vetor numérico do rosto)
    para cada imagem facial capturada do usuário.
    """
    pasta = os.path.join("faces", cpf)  # Pasta do usuário, ex: faces/12345678901
    os.makedirs(pasta, exist_ok=True)   # Cria a pasta se não existir

    for i, img in enumerate(imagens):
        # Gera o embedding usando o modelo ArcFace
        embedding = DeepFace.represent(img, model_name="ArcFace")[0]["embedding"]

        # Salva o embedding como arquivo .npy (NumPy)
        np.save(os.path.join(pasta, f"embedding_{i+1}.npy"), embedding)


# ---------------------------
# CLASSE CADASTRO
# ---------------------------
class Cadastro:
    """
    Classe responsável por cadastrar usuários e gerenciar:
    - Imagens faciais
    - Dados no banco de dados
    - Geração de embeddings
    """

    def __init__(self, nome, data_nascimento, cpf, imagens_faciais):
        self.nome = nome
        self.data_nascimento = data_nascimento
        self.cpf = cpf
        self.imagens_faciais = imagens_faciais  # Lista de imagens (rosto capturado)

    def cadastrar_usuario(self):
        """
        Salva múltiplas imagens faciais e dados do usuário no banco SQLite.
        Também gera embeddings para cada imagem.
        """
        # Cria pasta para as imagens do usuário
        pasta_cpf = os.path.join("faces", self.cpf)
        os.makedirs(pasta_cpf, exist_ok=True)

        imagem_paths = []  # Vai guardar os caminhos de todas as imagens capturadas

        # Salva cada imagem no disco
        for i, imagem in enumerate(self.imagens_faciais):
            imagem_path = os.path.join(pasta_cpf, f"{self.cpf}_{i+1}.jpg")

            # Redimensiona para 224x224 (compatível com ArcFace)
            imagem = cv2.resize(imagem, (224, 224))

            # Salva imagem em formato JPG
            cv2.imwrite(imagem_path, imagem)

            imagem_paths.append(imagem_path)

        # Junta todos os caminhos de imagens em uma única string separada por ";"
        imagens_str = ";".join(imagem_paths)

        # ---------------------------
        # BANCO DE DADOS
        # ---------------------------
        conn = sqlite3.connect("cadastro.db")  # Conecta (ou cria) banco SQLite
        cursor = conn.cursor()

        # Cria tabela de usuários se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                data_nascimento TEXT NOT NULL,
                cpf TEXT NOT NULL UNIQUE,
                imagem_facial TEXT NOT NULL
            )
        """)

        # Verifica se o CPF já existe no banco
        cursor.execute("SELECT * FROM usuarios WHERE cpf = ?", (self.cpf,))
        if cursor.fetchone():  # Se retornou algo, CPF já está cadastrado
            print("CPF já cadastrado!")
            conn.close()
            return False

        # Insere novo usuário
        cursor.execute("""
            INSERT INTO usuarios (nome, data_nascimento, cpf, imagem_facial)
            VALUES (?, ?, ?, ?)
        """, (self.nome, self.data_nascimento, self.cpf, imagens_str))

        conn.commit()  # Salva as mudanças
        conn.close()   # Fecha a conexão

        # ---------------------------
        # GERA E SALVA EMBEDDINGS
        # ---------------------------
        if self.imagens_faciais:
            salvar_embeddings(self.imagens_faciais, self.cpf)

        return True  # Cadastro concluído

    def validar_cpf(self):
        """
        Valida se o CPF possui exatamente 11 números.
        """
        return self.cpf.isdigit() and len(self.cpf) == 11

    def capturar_imagens_guiado(self):
        """
        Captura 4 imagens faciais com diferentes instruções usando a webcam.
        """
        instrucoes = [
            "Olhe para a câmera e mantenha o rosto centralizado.",
            "Vire levemente o rosto para a ESQUERDA.",
            "Vire levemente o rosto para a DIREITA.",
            "Aproxime o rosto da câmera."
        ]

        fotos_capturadas = []

        # Abre a webcam
        cap = cv2.VideoCapture(0)

        # Carrega o classificador Haar Cascade para detecção de rostos
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Percorre cada instrução
        for passo, instrucao in enumerate(instrucoes):
            messagebox.showinfo("Instrução", instrucao)  # Mostra instrução na tela

            while True:
                ret, frame = cap.read()  # Lê imagem da câmera
                if not ret:
                    messagebox.showerror("Erro", "Erro ao acessar a câmera.")
                    break

                # Converte para escala de cinza (necessário para Haar Cascade)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detecta rostos no frame
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                # Desenha retângulo no primeiro rosto detectado
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    break

                # Mostra a janela com a imagem ao vivo
                cv2.imshow("Cadastro Facial", frame)

                # Aguarda tecla pressionada
                key = cv2.waitKey(1)

                if key == ord(' '):  # Espaço → capturar foto
                    if len(faces) > 0:
                        # Pega as coordenadas do rosto detectado
                        x, y, w, h = faces[0]

                        # Recorta apenas o rosto
                        rosto = frame[y:y+h, x:x+w]

                        # Redimensiona para 224x224
                        imagem_tratada = cv2.resize(rosto, (224, 224))

                        # Adiciona à lista de fotos
                        fotos_capturadas.append(imagem_tratada)

                        # Mensagem de confirmação
                        messagebox.showinfo("Foto tirada", f"Foto {passo+1} capturada com sucesso!")
                        break
                    else:
                        # Se não detectou rosto, avisa
                        messagebox.showwarning("Atenção", "Nenhum rosto detectado. Tente novamente.")

            # Fecha a janela da câmera para essa etapa
            cv2.destroyAllWindows()

        # Libera a câmera
        cap.release()

        return fotos_capturadas
