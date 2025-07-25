# Sistema de Reconhecimento Facial

Este projeto é um sistema de reconhecimento facial que permite o cadastro, login e edição de usuários. A aplicação possui uma interface gráfica intuitiva, desenvolvida em Python com Tkinter, OpenCV e DeepFace.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
tc/
faces/
logins/
sistema-reconhecimento-facial-1/
├── src/
│   ├── main.py                   # Ponto de entrada da aplicação
│   ├── cadastro.py               # Classe para cadastro de usuários e captura de imagens
│   ├── login.py                  # Classe para autenticação facial
│   ├── editar.py                 # Classe para edição e exclusão de registros
│   ├── reconhecimento_facial.py  # Classe para reconhecimento facial
│   └── utils/
│       └── database.py           # Funções utilitárias para gerenciamento de banco de dados
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação do projeto
```

## Funcionalidades

- **Cadastro**: Permite cadastrar nome, data de nascimento, CPF e capturar imagens faciais (2 fotos + 2 tratadas) para reconhecimento futuro.
- **Login**: Login por reconhecimento facial, sem necessidade de digitar CPF. O sistema reconhece o usuário e libera o acesso se a acurácia for suficiente.
- **Edição**: Permite editar nome, data de nascimento e substituir as imagens faciais do usuário. Também é possível excluir o usuário, removendo seus dados e imagens do sistema.
- **Exclusão Completa**: Ao excluir um usuário, a pasta de imagens faciais correspondente ao CPF também é removida automaticamente.

## Instalação

1. Clone o repositório:
   ```bash
   git clone <URL do repositório>
   ```
2. Navegue até o diretório do projeto:
   ```bash
   cd sistema-reconhecimento-facial-1
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para iniciar a aplicação, execute o arquivo principal:
```bash
python src/main.py
```

Siga as instruções na interface gráfica para cadastrar, autenticar e editar usuários.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

---

**Observação:**  
O sistema utiliza webcam para captura e reconhecimento facial. Certifique-se de que sua webcam está funcionando corretamente e que você possui permissões para acessá-la.