from utils.database import get_user_by_cpf, update_user, delete_user
import cv2
import os
import shutil
from cadastro import salvar_embeddings, gerar_variacao_unica

class Editar:
    """Operações de busca, atualização e exclusão de usuários."""

    def buscar_usuario(self, cpf):
        """Busca usuário pelo CPF."""
        usuario = get_user_by_cpf(cpf)
        return usuario if usuario else None

    def atualizar_usuario(self, cpf, nome=None, data_nascimento=None):
        """Atualiza nome e/ou data de nascimento do usuário."""
        if self.buscar_usuario(cpf):
            update_user(cpf, nome=nome, data_nascimento=data_nascimento)
            return True
        return False

    def excluir_usuario(self, cpf):
        """Exclui usuário pelo CPF."""
        if self.buscar_usuario(cpf):
            delete_user(cpf)
            return True
        return False

    def recortar_rosto(self, frame):
        """Recorta o rosto da imagem capturada."""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w]
        return None

    def atualizar_fotos_faciais(self, cpf, novas_fotos):
        """
        Substitui todas as fotos faciais e embeddings do usuário pelo CPF.
        novas_fotos: lista de 4 imagens originais (cada uma será tratada com uma variação diferente)
        """
        pasta_cpf = os.path.join("faces", cpf)
        # Remove imagens e embeddings antigos
        if os.path.exists(pasta_cpf):
            shutil.rmtree(pasta_cpf)
        os.makedirs(pasta_cpf, exist_ok=True)

        imagens_para_salvar = []
        for i, imagem in enumerate(novas_fotos):
            imagem = cv2.resize(imagem, (224, 224))
            imagem_path = os.path.join(pasta_cpf, f"{cpf}_{2*i+1}.jpg")
            cv2.imwrite(imagem_path, imagem)
            imagens_para_salvar.append(imagem)
            # Gera e salva a variação única para cada foto
            variacao = gerar_variacao_unica(imagem, i)
            imagem_path_var = os.path.join(pasta_cpf, f"{cpf}_{2*i+2}.jpg")
            cv2.imwrite(imagem_path_var, variacao)
            imagens_para_salvar.append(variacao)

        # Salva novos embeddings
        salvar_embeddings(imagens_para_salvar, cpf)
        return True