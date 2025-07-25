import sqlite3
import shutil
import os

def connect_to_database():
    """Conecta ao banco de dados SQLite."""
    return sqlite3.connect('cadastro.db')

# ---------- Usuários ----------

def create_user_table():
    """Cria a tabela de usuários se não existir."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            data_nascimento TEXT NOT NULL,
            cpf TEXT NOT NULL UNIQUE,
            imagem_facial TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_user(nome, data_nascimento, cpf, imagem_facial):
    """Insere um novo usuário no banco."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO usuarios (nome, data_nascimento, cpf, imagem_facial)
        VALUES (?, ?, ?, ?)
    ''', (nome, data_nascimento, cpf, imagem_facial))
    conn.commit()
    conn.close()

def get_user_by_cpf(cpf):
    """Busca usuário pelo CPF."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM usuarios WHERE cpf = ?', (cpf,))
    user = cursor.fetchone()
    conn.close()
    return user

def update_user(cpf, nome=None, data_nascimento=None, imagem_facial=None):
    """Atualiza dados do usuário pelo CPF."""
    conn = connect_to_database()
    cursor = conn.cursor()
    if nome:
        cursor.execute('UPDATE usuarios SET nome = ? WHERE cpf = ?', (nome, cpf))
    if data_nascimento:
        cursor.execute('UPDATE usuarios SET data_nascimento = ? WHERE cpf = ?', (data_nascimento, cpf))
    if imagem_facial:
        cursor.execute('UPDATE usuarios SET imagem_facial = ? WHERE cpf = ?', (imagem_facial, cpf))
    conn.commit()
    conn.close()

def delete_user(cpf):
    """Exclui usuário pelo CPF e remove a pasta de imagens faciais."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM usuarios WHERE cpf = ?', (cpf,))
    conn.commit()
    conn.close()
    # Remove a pasta faces/CPF se existir
    pasta_cpf = os.path.join("faces", cpf)
    if os.path.exists(pasta_cpf):
        shutil.rmtree(pasta_cpf)

# ---------- Logins ----------

def create_login_table():
    """Cria a tabela de logins se não existir."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cpf TEXT NOT NULL,
            nome TEXT,
            acuracia INTEGER,
            data_hora TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_login(cpf, nome, acuracia, data_hora):
    """Insere um registro de login no banco."""
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO logins (cpf, nome, acuracia, data_hora)
        VALUES (?, ?, ?, ?)
    ''', (cpf, nome, acuracia, data_hora))
    conn.commit()
    conn.close()