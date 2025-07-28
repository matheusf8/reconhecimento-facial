from deepface import DeepFace
import cv2
import os
from utils.database import connect_to_database
from tkinter import messagebox

class Login:
    def tratar_imagem(self, imagem):
        """Redimensiona e prepara imagem para DeepFace."""
        print("Tratando imagem: redimensionamento para 224x224")
        return cv2.resize(imagem, (224, 224))

    def autenticar_facial(self):
        """
        Login facial: abre webcam, detecta rosto, autentica com DeepFace e mostra CPF.
        Não precisa digitar CPF, igual catraca.
        """
        cap = cv2.VideoCapture(0)
        imagem_capturada = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Não foi possível acessar a webcam.")
                break

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Reconhecimento Facial - Pressione 'q' para capturar", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    imagem_capturada = frame[y:y+h, x:x+w]
                else:
                    imagem_capturada = frame
                break

        cap.release()
        cv2.destroyAllWindows()

        if imagem_capturada is None:
            messagebox.showerror("Erro", "Nenhuma imagem capturada.")
            print("Nenhuma imagem capturada.")
            return None

        imagem_tratada = self.tratar_imagem(imagem_capturada)
        return self._autenticar_por_imagem(imagem_tratada, show_message=True)

    def autenticar_sem_cpf(self, frame):
        """
        Reconhece usuário apenas pela imagem facial, retorna CPF se reconhecido.
        Usado para integração com outros fluxos (ex: vídeo ao vivo).
        """
        imagem_tratada = self.tratar_imagem(frame)
        return self._autenticar_por_imagem(imagem_tratada, show_message=False)

    def autenticar_com_acuracia(self, frame, model_name='Dlib'):
        """
        Reconhece usuário pela imagem facial e retorna o CPF com a melhor acurácia.
        """
        imagem_tratada = self.tratar_imagem(frame)
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute('SELECT cpf, imagem_facial FROM usuarios')
        usuarios = cursor.fetchall()
        conn.close()
        melhor_score = None
        melhor_cpf = None
        for cpf, imagens_str in usuarios:
            imagens_cadastradas = imagens_str.split(";")
            for imagem_path in imagens_cadastradas:
                if not os.path.exists(imagem_path):
                    continue
                try:
                    resultado = DeepFace.verify(imagem_tratada, imagem_path, model_name=model_name)
                    score = 1 - resultado["distance"]
                    if resultado["verified"] and (melhor_score is None or score > melhor_score):
                        melhor_score = score
                        melhor_cpf = cpf
                except Exception:
                    continue
        return melhor_score, melhor_cpf

    def _autenticar_por_imagem(self, imagem_tratada, show_message=False):
        """
        Tenta autenticar a imagem tratada com todas as imagens cadastradas.
        Se show_message=True, mostra popups de acesso liberado/negado.
        """
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute('SELECT cpf, imagem_facial FROM usuarios')
        usuarios = cursor.fetchall()
        conn.close()

        print("Usando DeepFace com modelo ArcFace para autenticação facial!")
        for cpf, imagens_str in usuarios:
            imagens_cadastradas = imagens_str.split(";")
            for imagem_path in imagens_cadastradas:
                print(f"Comparando com imagem cadastrada: {imagem_path}")
                if not os.path.exists(imagem_path):
                    print(f"Imagem não encontrada: {imagem_path}")
                    continue
                try:
                    resultado = DeepFace.verify(imagem_tratada, imagem_path, model_name='ArcFace')
                    print(f"Resultado DeepFace: {resultado}")
                    if resultado["verified"] and resultado["distance"] < 0.4:
                        acuracia = max(0, int((1 - resultado["distance"]) * 100))
                        if show_message:
                            messagebox.showinfo("Acesso Liberado", f"Usuário reconhecido!\nCPF: {cpf}\nAcurácia: {acuracia}%")
                        print(f"Acesso liberado! Usuário reconhecido: CPF {cpf} | Acurácia: {acuracia}%")
                        return cpf
                except Exception as e:
                    print(f"Erro na verificação: {e}")
                    continue
        if show_message:
            messagebox.showwarning("Acesso Negado", "Nenhum usuário reconhecido.")
        print("Acesso negado! Nenhum usuário reconhecido.")
        return None

    def salvar_imagens_cadastradas(self, cpf, imagens_para_salvar):
        """
        Salva as imagens cadastradas no diretório 'imagens' e atualiza o banco de dados.
        """
        caminhos_salvos = []
        for i, img in enumerate(imagens_para_salvar):
            img_tratada = self.tratar_imagem(img)
            caminho = f"imagens/{cpf}_{i}.png"
            cv2.imwrite(caminho, img_tratada)
            caminhos_salvos.append(caminho)
        # Salve ";".join(caminhos_salvos) no campo imagem_facial do banco