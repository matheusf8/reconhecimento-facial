# ---------------------------------------------------
# IMPORTAÇÕES
# ---------------------------------------------------
import cv2
import os
import shutil
from utils.database import get_user_by_cpf, update_user, delete_user
from cadastro import salvar_embeddings, gerar_variacao_unica

# ---------------------------------------------------
# CLASSE EDITAR
# ---------------------------------------------------
class Editar:
    """
    Classe que centraliza as operações de:
    - Buscar um usuário
    - Atualizar dados cadastrais
    - Excluir usuários
    - Atualizar fotos faciais e embeddings
    """

    def buscar_usuario(self, cpf):
        """
        Busca usuário no banco de dados usando CPF.
        Retorna os dados do usuário se encontrado, ou None se não existir.
        """
        usuario = get_user_by_cpf(cpf)
        return usuario if usuario else None

    def atualizar_usuario(self, cpf, nome=None, data_nascimento=None):
        """
        Atualiza nome e/ou data de nascimento de um usuário no banco.
        Se o usuário não existir, retorna False.
        """
        if self.buscar_usuario(cpf):
            update_user(cpf, nome=nome, data_nascimento=data_nascimento)
            return True
        return False

    def excluir_usuario(self, cpf):
        """
        Remove o usuário do banco de dados.
        Retorna True se foi excluído, False se não existir.
        """
        if self.buscar_usuario(cpf):
            delete_user(cpf)
            return True
        return False

    def recortar_rosto(self, frame):
        """
        Localiza e recorta o rosto detectado na imagem (frame).
        Retorna apenas a região do rosto ou None se nenhum rosto for encontrado.
        """
        # Carrega o classificador Haar Cascade para detectar rostos
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Converte para escala de cinza para melhorar a detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostos na imagem
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Se pelo menos um rosto foi encontrado, retorna o primeiro
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w]

        # Se não encontrou rosto, retorna None
        return None

    def atualizar_fotos_faciais(self, cpf, novas_fotos):
        """
        Substitui TODAS as fotos faciais e embeddings do usuário.
        novas_fotos: lista com 4 imagens originais (rosto do usuário).
        Para cada imagem original, também é gerada uma variação artificial
        para aumentar a robustez do reconhecimento facial.
        """
        # Define a pasta de armazenamento das fotos do usuário
        pasta_cpf = os.path.join("faces", cpf)

        # Remove pasta antiga se existir (imagens e embeddings antigos)
        if os.path.exists(pasta_cpf):
            shutil.rmtree(pasta_cpf)

        # Cria pasta limpa para salvar as novas imagens
        os.makedirs(pasta_cpf, exist_ok=True)

        # Lista que armazenará todas as imagens (originais + variações)
        imagens_para_salvar = []

        for i, imagem in enumerate(novas_fotos):
            # Redimensiona imagem para 224x224 (compatível com ArcFace)
            imagem = cv2.resize(imagem, (224, 224))

            # Salva a imagem original
            imagem_path = os.path.join(pasta_cpf, f"{cpf}_{2*i+1}.jpg")
            cv2.imwrite(imagem_path, imagem)
            imagens_para_salvar.append(imagem)

            # Gera uma variação única da imagem original
            variacao = gerar_variacao_unica(imagem, i)

            # Salva a variação
            imagem_path_var = os.path.join(pasta_cpf, f"{cpf}_{2*i+2}.jpg")
            cv2.imwrite(imagem_path_var, variacao)
            imagens_para_salvar.append(variacao)

        # Gera e salva embeddings para TODAS as imagens
        salvar_embeddings(imagens_para_salvar, cpf)

        return True
