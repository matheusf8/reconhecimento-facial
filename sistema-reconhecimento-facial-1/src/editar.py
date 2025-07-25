from utils.database import get_user_by_cpf, update_user, delete_user
import cv2

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