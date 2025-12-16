import json
import getpass
from pathlib import Path
import bcrypt

USERS_FILE = Path("secrets/users.json")
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

def gerar_hash_senha(senha: str) -> str:
    return bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

def carregar() -> dict:
    if USERS_FILE.exists():
        try:
            data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def salvar(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

def listar(users: dict) -> None:
    nomes = sorted(users.keys())
    print(f"\nTotal de usuários: {len(nomes)}")
    for u in nomes:
        print(f" - {u}")
    print()

def adicionar(users: dict) -> None:
    usuario = input("Novo usuário: ").strip()
    if not usuario:
        print("Usuário vazio. Cancelado.\n")
        return
    if usuario in users:
        print("Usuário já existe. Use 'Resetar senha' se quiser trocar.\n")
        return
    senha = getpass.getpass("Senha: ")
    users[usuario] = gerar_hash_senha(senha)
    salvar(users)
    print("✅ Usuário criado.\n")

def resetar(users: dict) -> None:
    usuario = input("Usuário para resetar senha: ").strip()
    if usuario not in users:
        print("Usuário não encontrado.\n")
        return
    senha = getpass.getpass("Nova senha: ")
    users[usuario] = gerar_hash_senha(senha)
    salvar(users)
    print("✅ Senha atualizada.\n")

def remover(users: dict) -> None:
    usuario = input("Usuário para remover: ").strip()
    if usuario not in users:
        print("Usuário não encontrado.\n")
        return
    conf = input(f"Tem certeza que deseja remover '{usuario}'? (s/N): ").strip().lower()
    if conf == "s":
        users.pop(usuario, None)
        salvar(users)
        print("✅ Usuário removido.\n")
    else:
        print("Cancelado.\n")

def main():
    while True:
        users = carregar()
        print("=== Gerenciador de Usuários (Nathal.IA) ===")
        print("1) Listar usuários")
        print("2) Adicionar usuário")
        print("3) Resetar senha")
        print("4) Remover usuário")
        print("5) Sair")
        op = input("Escolha: ").strip()

        if op == "1":
            listar(users)
        elif op == "2":
            adicionar(users)
        elif op == "3":
            resetar(users)
        elif op == "4":
            remover(users)
        elif op == "5":
            print("Saindo.")
            break
        else:
            print("Opção inválida.\n")

if __name__ == "__main__":
    main()
