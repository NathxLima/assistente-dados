import os
import fitz  # PyMuPDF
import json

# Caminho da pasta com os PDFs
PASTA_PDF = "docs/a_vaga_SEBRAE"
SAIDA_JSONL = "data/sebrae_finetune_corrigido.jsonl"

def extrair_texto_pdf(caminho_pdf):
    texto_total = ""
    with fitz.open(caminho_pdf) as doc:
        for pagina in doc:
            texto_total += pagina.get_text()
    return texto_total.strip()

# Gera exemplos com role estruturado
def gerar_exemplos(texto):
    exemplos = []
    blocos = texto.split("\n\n")  # separa por blocos maiores de conteúdo
    for bloco in blocos:
        bloco = bloco.strip()
        if len(bloco) > 100:
            exemplo = {
                "messages": [
                    {"role": "user", "content": f"Resuma esse trecho como se estivesse se preparando para uma entrevista técnica para o SEBRAE:\n\n{bloco}"},
                    {"role": "assistant", "content": f"[RESPOSTA RESUMIDA AQUI]"}  # Substituiremos manual ou via GPT
                ]
            }
            exemplos.append(exemplo)
    return exemplos

def main():
    todos_exemplos = []
    for nome_arquivo in os.listdir(PASTA_PDF):
        if nome_arquivo.endswith(".pdf"):
            caminho_pdf = os.path.join(PASTA_PDF, nome_arquivo)
            print(f"Extraindo de: {caminho_pdf}")
            texto = extrair_texto_pdf(caminho_pdf)
            exemplos = gerar_exemplos(texto)
            todos_exemplos.extend(exemplos)

    # Salvando os exemplos em JSONL
    with open(SAIDA_JSONL, "w", encoding="utf-8") as f:
        for exemplo in todos_exemplos:
            json.dump(exemplo, f, ensure_ascii=False)
            f.write("\n")
    print(f"\n✅ Arquivo gerado com {len(todos_exemplos)} exemplos: {SAIDA_JSONL}")

if __name__ == "__main__":
    main()

