from tkinter import Tk, Label, Button, Entry, messagebox, Toplevel, Frame, Listbox
from cadastro import Cadastro
from utils.database import update_user, delete_user, get_user_by_cpf, connect_to_database, insert_login, create_login_table
from login import Login
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from deepface import DeepFace
import numpy as np
import datetime
import warnings

warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime avisos e infos do TensorFlow

class SistemaReconhecimentoFacial:
    def __init__(self, master):
        self.master = master
        master.title("Sistema de Reconhecimento Facial")
        master.geometry("550x400")
        master.configure(bg="#f0f0f0")

        self.frame_menu = Frame(master, bg="#f0f0f0")
        self.frame_menu.pack(expand=True, fill="both")

        Label(self.frame_menu, text="Bem-vindo ao Sistema de Reconhecimento Facial",
              font=("Arial", 15, "bold"), bg="#f0f0f0", wraplength=540).pack(pady=20)

        Button(self.frame_menu, text="Login", width=20, height=2, command=self.abrir_login_facial,
               bg="#4caf50", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        Button(self.frame_menu, text="Cadastro", width=20, height=2, command=self.mostrar_cadastro,
               bg="#2196f3", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        Button(self.frame_menu, text="Editar", width=20, height=2, command=self.mostrar_editar,
               bg="#ff9800", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        Button(self.frame_menu, text="Sair", width=20, height=2, command=master.quit,
               bg="#f44336", fg="white", font=("Arial", 12, "bold")).pack(pady=5)

        self.frame_cadastro = Frame(master, bg="#f0f0f0")
        self.frame_editar = Frame(master, bg="#f0f0f0")

    def mostrar_menu(self):
        self.frame_cadastro.pack_forget()
        self.frame_editar.pack_forget()
        self.frame_menu.pack(expand=True, fill="both")

    # --- Cadastro ---
    def mostrar_cadastro(self):
        self.frame_menu.pack_forget()
        self.frame_editar.pack_forget()
        self.frame_cadastro.pack(expand=True, fill="both")
        self.construir_cadastro()

    def construir_cadastro(self):
        for widget in self.frame_cadastro.winfo_children():
            widget.destroy()
        self.master.geometry("680x570")
        Label(self.frame_cadastro, text="Cadastro de Usuário", font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=20)

        Label(self.frame_cadastro, text="Nome:", font=("Arial", 14), bg="#f0f0f0").pack(pady=5)
        entry_nome = Entry(self.frame_cadastro, font=("Arial", 14), width=30)
        entry_nome.pack(pady=5)

        Label(self.frame_cadastro, text="Data de Nascimento:", font=("Arial", 14), bg="#f0f0f0").pack(pady=5)
        entry_data = Entry(self.frame_cadastro, font=("Arial", 14), width=30)
        entry_data.pack(pady=5)

        Label(self.frame_cadastro, text="CPF:", font=("Arial", 14), bg="#f0f0f0").pack(pady=5)
        entry_cpf = Entry(self.frame_cadastro, font=("Arial", 14), width=30)
        entry_cpf.pack(pady=5)

        self.imagens_faciais_capturadas = []
        foto_status_label = Label(self.frame_cadastro, text="Fotos salvas: Não", font=("Arial", 13), bg="#f0f0f0", fg="red")
        foto_status_label.pack(pady=10)

        def gerar_variacoes(imagem):
            imagens = [imagem]
            img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            img_eq = cv2.equalizeHist(img_cinza)
            img_final = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
            imagens.append(img_final)
            return imagens

        def abrir_camera():
            # Aviso de iluminação antes de abrir a câmera
            messagebox.showinfo("Dica de Cadastro", "Tenha boa iluminação para a foto!\nEvite sombras e mantenha o rosto centralizado.")

            camera_window = Toplevel(self.frame_cadastro)
            camera_window.title("Capturar Rosto")
            camera_window.geometry("600x600")
            camera_window.configure(bg="#f0f0f0")

            frame_video = Frame(camera_window, bg="#f0f0f0", height=400)
            frame_video.pack(fill="x")
            frame_video.pack_propagate(False)
            l_video = Label(frame_video)
            l_video.pack(expand=True)

            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            fotos_capturadas = []

            # Defina as variáveis locais para o escopo do cadastro
            autenticado = [False]
            ultimo_frame = [None]
            lock = threading.Lock()
            ultimo_rosto_detectado = [0]
            mensagem_posicionamento = [None]
            tempo_espera = 1.2  # segundos para a pessoa se posicionar

            def capturar_automatico():
                count = 0
                while count < 2:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        rosto = frame[y:y+h, x:x+w]
                        rosto = cv2.resize(rosto, (224, 224))
                        for img_var in gerar_variacoes(rosto):
                            fotos_capturadas.append(img_var)
                        count += 1
                        l_video.after(1, lambda: l_video.config(text=f"Capturando: {count}/2"))
                    camera_window.update()
                    time.sleep(0.2)
                cap.release()
                camera_window.destroy()
                self.imagens_faciais_capturadas = fotos_capturadas.copy()
                foto_status_label.config(text="Fotos salvas: Sim", fg="green")

            def mostrar_video():
                ret, frame = cap.read()
                if ret:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        cor = (0, 255, 0) if autenticado[0] else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
                        break
                    with lock:
                        ultimo_frame[0] = frame.copy()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    l_video.imgtk = imgtk
                    l_video.configure(image=imgtk)
                l_video.after(100, mostrar_video)

            Button(camera_window, text="Iniciar Captura Automática", command=lambda: threading.Thread(target=capturar_automatico).start(),
                   bg="#4caf50", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
            Button(camera_window, text="Fechar", command=lambda: [cap.release(), camera_window.destroy()],
                   bg="#f44336", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

            mostrar_video()

        Button(self.frame_cadastro, text="Abrir Câmera", command=abrir_camera, bg="#2196f3", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)

        def salvar_cadastro():
            nome = entry_nome.get()
            data_nascimento = entry_data.get()
            cpf = entry_cpf.get()
            if hasattr(self, "imagens_faciais_capturadas") and len(self.imagens_faciais_capturadas) >= 4:
                cadastro = Cadastro(nome, data_nascimento, cpf, self.imagens_faciais_capturadas)
                if cadastro.validar_cpf():
                    cadastro.cadastrar_usuario()
                    messagebox.showinfo("Cadastro", "Cadastro realizado com sucesso!")
                    self.imagens_faciais_capturadas = []
                    self.frame_cadastro.destroy()
                    self.mostrar_menu()
                else:
                    messagebox.showerror("Erro", "CPF inválido.")
            else:
                messagebox.showerror("Erro", "Capture 2 imagens faciais antes de salvar.")

        Button(self.frame_cadastro, text="Salvar Cadastro", command=salvar_cadastro, bg="#4caf50", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)
        Button(self.frame_cadastro, text="← Voltar", command=self.mostrar_menu, bg="#2196f3", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)

    # --- Edição ---
    def mostrar_editar(self):
        self.frame_menu.pack_forget()
        self.frame_cadastro.pack_forget()
        self.frame_editar.pack(expand=True, fill="both")
        self.construir_editar()

    def construir_editar(self):
        for widget in self.frame_editar.winfo_children():
            widget.destroy()
        self.master.geometry("600x500")
        Label(self.frame_editar, text="Editar Usuário", font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=20)

        Label(self.frame_editar, text="Digite o CPF:", font=("Arial", 14), bg="#f0f0f0").pack(pady=5)
        entry_cpf = Entry(self.frame_editar, font=("Arial", 14), width=30)
        entry_cpf.pack(pady=10)

        def alterar():
            cpf = entry_cpf.get()
            usuario = get_user_by_cpf(cpf)
            if usuario:
                alterar_window = Toplevel(self.frame_editar)
                alterar_window.title("Alterar Usuário")
                alterar_window.geometry("400x400")
                alterar_window.configure(bg="#f0f0f0")

                Label(alterar_window, text="Novo Nome:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
                entry_nome = Entry(alterar_window, font=("Arial", 12))
                entry_nome.insert(0, usuario[1])
                entry_nome.pack(pady=5)

                Label(alterar_window, text="Nova Data de Nascimento:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
                entry_data = Entry(alterar_window, font=("Arial", 12))
                entry_data.insert(0, usuario[2])
                entry_data.pack(pady=5)

                novas_imagens_faciais = []

                def gerar_variacoes(imagem):
                    imagens = [imagem]
                    img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
                    img_eq = cv2.equalizeHist(img_cinza)
                    img_final = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
                    imagens.append(img_final)
                    return imagens

                def salvar_novas_fotos():
                    camera_window = Toplevel(alterar_window)
                    camera_window.title("Salvar Novas Fotos")
                    camera_window.geometry("500x400")
                    camera_window.configure(bg="#f0f0f0")

                    frame_video = Frame(camera_window, bg="#f0f0f0", height=300)
                    frame_video.pack(fill="x")
                    frame_video.pack_propagate(False)
                    l_video = Label(frame_video)
                    l_video.pack(expand=True)

                    cap = cv2.VideoCapture(0)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    fotos_capturadas = []

                    def capturar_automatico():
                        count = 0
                        while count < 2:
                            ret, frame = cap.read()
                            if not ret:
                                continue
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            if len(faces) > 0:
                                x, y, w, h = faces[0]
                                rosto = frame[y:y+h, x:x+w]
                                rosto = cv2.resize(rosto, (224, 224))
                                for img_var in gerar_variacoes(rosto):
                                    fotos_capturadas.append(img_var)
                                count += 1
                                l_video.after(1, lambda: l_video.config(text=f"Capturando: {count}/2"))
                            camera_window.update()
                            time.sleep(0.2)
                        cap.release()
                        camera_window.destroy()
                        # Apaga as imagens antigas e salva as novas
                        pasta_cpf = os.path.join("faces", usuario[3])
                        import shutil
                        if os.path.exists(pasta_cpf):
                            shutil.rmtree(pasta_cpf)
                        os.makedirs(pasta_cpf, exist_ok=True)
                        imagem_paths = []
                        for i, imagem in enumerate(fotos_capturadas):
                            imagem_path = os.path.join(pasta_cpf, f"{usuario[3]}_{i+1}.jpg")
                            cv2.imwrite(imagem_path, imagem)
                            imagem_paths.append(imagem_path)
                        novas_imagens_faciais.clear()
                        novas_imagens_faciais.extend(imagem_paths)
                        messagebox.showinfo("Captura", "Novas fotos salvas com sucesso!")

                    Button(camera_window, text="Iniciar Captura Automática", command=lambda: threading.Thread(target=capturar_automatico).start(),
                           bg="#4caf50", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
                    Button(camera_window, text="Fechar", command=lambda: [cap.release(), camera_window.destroy()],
                           bg="#f44336", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

                    def mostrar_video():
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            imgtk = ImageTk.PhotoImage(image=img)
                            l_video.imgtk = imgtk
                            l_video.configure(image=imgtk)
                        l_video.after(10, mostrar_video)

                    mostrar_video()

                Button(alterar_window, text="Salvar Novas Fotos", command=salvar_novas_fotos, bg="#2196f3", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

                def salvar_alteracao():
                    novo_nome = entry_nome.get()
                    nova_data = entry_data.get()
                    update_user(usuario[3], nome=novo_nome, data_nascimento=nova_data)
                    if novas_imagens_faciais:
                        imagens_str = ";".join(novas_imagens_faciais)
                        update_user(usuario[3], imagem_facial=imagens_str)
                    messagebox.showinfo("Alterar", "Dados alterados com sucesso!")
                    alterar_window.destroy()

                Button(alterar_window, text="Salvar", command=salvar_alteracao, bg="#4caf50", fg="white", font=("Arial", 12, "bold")).pack(pady=15)
            else:
                messagebox.showerror("Erro", "Usuário não encontrado!")

        def excluir():
            cpf = entry_cpf.get()
            usuario = get_user_by_cpf(cpf)
            if usuario:
                delete_user(cpf)
                messagebox.showinfo("Excluir", "Usuário excluído com sucesso!")
            else:
                messagebox.showerror("Erro", "Usuário não encontrado!")

        Button(self.frame_editar, text="Alterar", command=alterar, bg="#ff9800", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)
        Button(self.frame_editar, text="Excluir", command=excluir, bg="#f44336", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)

        def buscar_todos():
            lista_window = Toplevel(self.frame_editar)
            lista_window.title("Usuários Cadastrados")
            lista_window.geometry("400x400")
            lista_window.configure(bg="#f0f0f0")

            conn = connect_to_database()
            cursor = conn.cursor()
            cursor.execute("SELECT id, nome, cpf FROM usuarios")
            usuarios = cursor.fetchall()
            conn.close()

            Label(lista_window, text="Selecione um usuário:", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)
            listbox = Listbox(lista_window, font=("Arial", 12), width=40)
            listbox.pack(pady=10, expand=True)

            for usuario in usuarios:
                listbox.insert("end", f"{usuario[1]} - CPF: {usuario[2]}")

            def on_select(event):
                idx = listbox.curselection()
                if not idx:
                    return
                usuario = usuarios[idx[0]]
                opcoes_window = Toplevel(lista_window)
                opcoes_window.title("Opções do Usuário")
                opcoes_window.geometry("300x200")
                opcoes_window.configure(bg="#f0f0f0")

                Label(opcoes_window, text=f"{usuario[1]}\nCPF: {usuario[2]}", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=20)

                def excluir_selecionado():
                    delete_user(usuario[2])
                    messagebox.showinfo("Excluir", "Usuário excluído com sucesso!")
                    opcoes_window.destroy()
                    lista_window.destroy()

                def alterar_selecionado():
                    entry_cpf.delete(0, "end")
                    entry_cpf.insert(0, usuario[2])
                    opcoes_window.destroy()
                    lista_window.destroy()
                    alterar()

                Button(opcoes_window, text="Excluir", command=excluir_selecionado, bg="#f44336", fg="white", font=("Arial", 12, "bold"), width=12).pack(pady=10)
                Button(opcoes_window, text="Alterar", command=alterar_selecionado, bg="#ff9800", fg="white", font=("Arial", 12, "bold"), width=12).pack(pady=10)

            listbox.bind("<<ListboxSelect>>", on_select)

        Button(self.frame_editar, text="Buscar Todos", command=buscar_todos, bg="#2196f3", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)
        Button(self.frame_editar, text="← Voltar", command=self.mostrar_menu, bg="#2196f3", fg="white", font=("Arial", 14, "bold"), width=18).pack(pady=10)

    # --- Login Facial Moderno ---
    def abrir_login_facial(self):
        login_window = Toplevel(self.master)
        login_window.title("Reconhecimento Facial")
        login_window.geometry("700x500")
        login_window.configure(bg="#222")

        # Remova o frame_top e o Label "Login Facial"
        # frame_top = Frame(login_window, bg="#222", height=100)
        # frame_top.pack(fill="x")
        # Label(frame_top, text="Login Facial", font=("Arial", 36, "bold"), fg="#00ff88", bg="#222").pack(pady=30)

        frame_video = Frame(login_window, bg="#222", width=500, height=300)
        frame_video.pack(expand=True)
        frame_video.pack_propagate(False)
        l_video = Label(frame_video, bg="#222")
        l_video.pack(expand=True)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        autenticado = [False]
        ultima_verificacao = [time.time()]
        ultimo_frame = [None]
        lock = threading.Lock()

        def carregar_embeddings():
            embeddings = {}
            if not os.path.exists("faces"):
                return embeddings
            for cpf in os.listdir("faces"):
                emb_path = os.path.join("faces", cpf, "embedding.npy")
                if os.path.exists(emb_path):
                    embeddings[cpf] = np.load(emb_path)
            return embeddings

        embeddings_cadastrados = carregar_embeddings()

        ultimo_rosto_presente = [None]
        tempo_espera = 4.0  # segundos

        # Mensagem fixa, agora em verde e centralizada
        label_posicione = Label(
            login_window,
            text="Posicione o rosto",
            font=("Arial", 22, "bold"),
            fg="#00ff88",
            bg="#222"
        )
        label_posicione.place(relx=0.5, rely=0.12, anchor="center")

        def mostrar_video():
            if not autenticado[0]:
                ret, frame = cap.read()
                if ret:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    agora = time.time()
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        cor = (0, 255, 0) if autenticado[0] else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
                        if ultimo_rosto_presente[0] is None:
                            ultimo_rosto_presente[0] = agora
                    else:
                        ultimo_rosto_presente[0] = None
                    with lock:
                        ultimo_frame[0] = frame.copy()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    l_video.imgtk = imgtk
                    l_video.configure(image=imgtk)
                l_video.after(100, mostrar_video)

        def reconhecimento_em_background():
            tempo_inicio = time.time()
            tempo_limite = 30  # segundos
            while not autenticado[0]:
                time.sleep(0.1)
                if time.time() - ultima_verificacao[0] > 1.0:
                    with lock:
                        frame = ultimo_frame[0].copy() if ultimo_frame[0] is not None else None
                    if frame is not None and ultimo_rosto_presente[0] is not None:
                        if time.time() - ultimo_rosto_presente[0] >= tempo_espera:
                            try:
                                embedding_atual = DeepFace.represent(frame, model_name="ArcFace")[0]["embedding"]
                                for cpf, emb_cad in embeddings_cadastrados.items():
                                    dist = np.linalg.norm(np.array(emb_cad) - np.array(embedding_atual))
                                    if dist < 10:
                                        autenticado[0] = True
                                        cap.release()
                                        acuracia = max(0, int((1 - dist/10) * 100))
                                        usuario = get_user_by_cpf(cpf)
                                        nome = usuario[1] if usuario else "Usuário"
                                        login_window.after(0, lambda: self.mostrar_acesso_liberado(cpf, nome, acuracia))
                                        login_window.after(0, login_window.destroy)
                                        break
                            except Exception as e:
                                print("Erro:", e)
                    ultima_verificacao[0] = time.time()
                if time.time() - tempo_inicio > tempo_limite:
                    cap.release()
                    login_window.after(0, lambda: messagebox.showwarning("Atenção", "Nenhum rosto reconhecido."))
                    login_window.after(0, login_window.destroy)
                    break
        threading.Thread(target=reconhecimento_em_background, daemon=True).start()
        mostrar_video()

    def mostrar_acesso_liberado(self, cpf, nome, acuracia):
        self.registrar_login(cpf, nome, acuracia)
        print(f"LOGIN: {nome} | CPF: {cpf} | {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Acurácia: {acuracia}%")
        acesso_window = Toplevel(self.master)
        acesso_window.title("Acesso Liberado")
        acesso_window.geometry("600x450")
        acesso_window.configure(bg="#222")

        frame_central = Frame(acesso_window, bg="#222")
        frame_central.place(relx=0.5, rely=0.5, anchor="center")

        label_icone = Label(frame_central, text="🔓", font=("Arial", 80), fg="#00ff88", bg="#222")
        label_icone.pack(pady=(0, 10))

        label_msg = Label(frame_central, text="Acesso Liberado!", font=("Arial", 26, "bold"), fg="#00ff88", bg="#222")
        label_msg.pack(pady=(0, 10))

        label_cpf = Label(frame_central, text=f"CPF: {cpf}", font=("Arial", 18), fg="#fff", bg="#222")
        label_cpf.pack(pady=(0, 5))

        label_nome = Label(frame_central, text=f"Bem-vindo, {nome}!", font=("Arial", 18, "bold"), fg="#00ff88", bg="#222")
        label_nome.pack(pady=(0, 15))

        label_acuracia = Label(frame_central, text=f"Acurácia: {acuracia}%", font=("Arial", 16), fg="#fff", bg="#222")
        label_acuracia.pack(pady=(0, 10))

        barra = Frame(frame_central, bg="#444", width=220, height=18)
        barra.pack(pady=(0, 20))
        progresso = Frame(barra, bg="#00ff88", width=0, height=18)
        progresso.place(x=0, y=0)

        btn_fechar = Button(frame_central, text="Fechar", command=lambda: [acesso_window.destroy(), self.mostrar_menu()],
                            font=("Arial", 14, "bold"), bg="#00ff88", fg="#222", relief="flat", width=12)
        btn_fechar.pack()

        def animacao():
            for i in range(0, 221, 22):
                progresso.config(width=i)
                cor = "#00ff88" if i % 44 == 0 else "#00e0ff"
                label_icone.config(fg=cor)
                label_msg.config(fg=cor)
                acesso_window.update()
                time.sleep(0.18)
            btn_fechar.config(state="normal")

        btn_fechar.config(state="disabled")
        threading.Thread(target=animacao).start()

    def tratar_imagem(self, imagem):
        if imagem is None:
            raise ValueError("Imagem inválida para tratamento.")
        return cv2.resize(imagem, (224, 224))

    def autenticar_com_acuracia(self, frame, model_name='ArcFace'):
        try:
            print("Iniciando autenticação facial...")
            imagem_tratada = self.tratar_imagem(frame)
            conn = connect_to_database()
            cursor = conn.cursor()
            cursor.execute('SELECT cpf, imagem_facial FROM usuarios')
            usuarios = cursor.fetchall()
            print(f"Usuários encontrados no banco: {len(usuarios)}")
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
        except Exception as e:
            print(f"Erro geral na autenticação: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro geral", f"{e}")
            return None, None

    def registrar_login(self, cpf, nome, acuracia):
        # Salva no banco
        agora = datetime.datetime.now()
        data_hora = agora.strftime("%d/%m/%Y %H:%M:%S")
        insert_login(cpf, nome, acuracia, data_hora)

        # Salva em arquivo por dia
        pasta_log = "logins"
        if not os.path.exists(pasta_log):
            os.makedirs(pasta_log)
        nome_arquivo = agora.strftime("%d-%m-%Y") + ".txt"
        caminho_log = os.path.join(pasta_log, nome_arquivo)
        with open(caminho_log, "a", encoding="utf-8") as f:
            f.write(f"{data_hora} | CPF: {cpf} | Nome: {nome} | Acurácia: {acuracia}%\n")

if __name__ == "__main__":
    create_login_table()  # Garante que a tabela de logins existe
    root = Tk()
    sistema = SistemaReconhecimentoFacial(root)
    print("Sistema iniciado!")
    root.mainloop()
    print("Programa finalizado!")