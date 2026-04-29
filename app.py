# IMPORTAR BIBLIOTECAS
import os
import re

import chromadb
import ollama

from pypdf import PdfReader
from langchain_text_splitters import (RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter)

# =========================================================
# 1 - LEITURA DOS PDFs E EXTRAÇÃO DE TEXTO
# =========================================================

pdf_directory = 'pdfs'  # Pasta onde estão os PDFs

def load_documents_from_directory(directory):
    documents = []  # Lista onde serão guardados os documentos processados

    # Percorre todos os ficheiros na pasta indicada
    for file in os.listdir(directory):

        # Apenas processa ficheiros PDF
        if file.endswith('.pdf'):

            # Cria o caminho completo do ficheiro
            file_path = os.path.join(directory, file)

            # Abre o PDF com o PyPDF
            reader = PdfReader(file_path)

            text = ''  # Variável para acumular o texto do PDF

            # Percorre todas as páginas do PDF
            for page in reader.pages:

                # Extrai o texto de cada página e junta ao texto total
                text += page.extract_text()

            # =========================================================
            # LIMPEZA DO TEXTO EXTRAÍDO DO PDF
            # =========================================================

            # Remove múltiplos espaços e normaliza para 1 espaço
            text = re.sub(r'\s+', ' ', text)

            # Separa números colados a letras (ex: "33Norte" -> "33 Norte")
            text = re.sub(r'(\d)([A-Za-zÀ-ÿ])', r'\1 \2', text)

            # Remove pares de números típicos de páginas (ex: "38 39")
            text = re.sub(r'\b\d{1,3}\s+\d{1,3}\b', ' ', text)

            # Substitui quebras de linha simples por espaço
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

            # Limpeza final de espaços
            text = re.sub(r'\s+', ' ', text).strip()

            # Guarda o documento processado
            documents.append({
                'file': file,   # Nome do ficheiro
                'text': text    # Texto limpo extraído
            })

    return documents  # Devolve todos os documentos

# Carrega todos os PDFs da pasta
documents = load_documents_from_directory(pdf_directory)

# Mantém apenas o texto (ignora nome do ficheiro)
documents = [doc['text'] for doc in documents]

# =========================================================
# 2 - CHUNKING (DIVISÃO DO TEXTO EM PEDAÇOS)
# =========================================================

# Splitter baseado em regras de texto (parágrafos, frases, etc.)
spliter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    chunk_size=1000,     # tamanho máximo de cada chunk
    chunk_overlap=200,   # sobreposição entre chunks
)

# Junta todos os documentos num só texto e divide em chunks
text_chunks = spliter.split_text("\n\n".join(documents))

print(f"Total de pedaços de texto gerados: {len(text_chunks)}")

# =========================================================
# 3 - SPLITTER BASEADO EM TOKENS (para embeddings melhores)
# =========================================================

token_splitter = SentenceTransformersTokenTextSplitter(
    model_name="all-MiniLM-L6-v2",  # modelo usado para tokenização
    tokens_per_chunk=256,           # tamanho máximo por chunk em tokens
    chunk_overlap=50                # sobreposição entre chunks
)

token_chuncks = []  # lista final de chunks em tokens

# Converte cada chunk em versão otimizada por tokens
for text in text_chunks:
    token_chuncks += token_splitter.split_text(text)

# =========================================================
# 4 - CRIAÇÃO DA BASE DE DADOS VETORIAL (CHROMA DB)
# =========================================================

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Função que transforma texto em vetores (embeddings)
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Cria ou abre base de dados persistente
client = chromadb.PersistentClient()

# Cria ou obtém coleção onde os embeddings serão guardados
chroma_collection = client.get_or_create_collection(
    name="colecao",
    embedding_function=embedding_function
)

# IDs únicos para cada chunk
ids = [str(i) for i in range(len(token_chuncks))]

# Adiciona os chunks à base vetorial
chroma_collection.add(
    ids=ids,
    documents=token_chuncks
)

count = chroma_collection.count()
print(f"Total de pedaços de texto armazenados: {count}")

# =========================================================
# 5 - REESCRITA DA PERGUNTA COM CONTEXTO (RAG melhorado)
# =========================================================

def rewrite_question_with_history(question, model):
    """
    Reescreve a pergunta usando histórico da conversa.
    Objetivo: tornar a pergunta independente de contexto anterior.
    """

    messages = [
        {
            "role": "system",
            "content": "Reescreve a pergunta do utilizador para ficar independente. Usa histórico da conversa. Não respondas, apenas reformula."
        }
    ]

    # adiciona histórico da conversa
    messages.extend(chat_history)

    # pergunta atual do utilizador
    messages.append({
        "role": "user",
        "content": question
    })

    # chama modelo LLM (Ollama)
    reponse = ollama.chat(
        model=model,
        messages=messages
    )

    return reponse["message"]["content"]

# =========================================================
# 6 - QUERY EXPANSION (HYDE - geração de resposta hipotética)
# =========================================================

def augment_query_generation(query, model):
    """
    Gera uma resposta hipotética para melhorar a pesquisa semântica.
    Técnica: HYDE (Hypothetical Document Embeddings)
    """

    prompt = f"""Você é um assistente turistico especializado em Portugal.
Fornece um exemplo de resposta para a perguntada do utilizador."""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )

    return response["message"]["content"]

# =========================================================
# 7 - PESQUISA SEMÂNTICA NA BASE DE DADOS (CHROMA)
# =========================================================

def chroma_query(query):
    """
    Faz pesquisa semântica na base vetorial.
    Retorna os documentos mais relevantes.
    """

    results = chroma_collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "embeddings"]
    )

    return results

# =========================================================
# 8 - GERAÇÃO DA RESPOSTA FINAL (RAG + LLM)
# =========================================================

chat_history = []  # histórico da conversa

def generate_final_answer(question, retrieved_docs, model):

    # Junta documentos relevantes num único contexto
    context = "\n\n".join(retrieved_docs)

    messages = [
        {
            "role": "system",
            "content": "Você é um assistente turístico especializado em Portugal. Responda à pergunta do utilizador utilizando os documentos fornecidos."
        }
    ]

    # adiciona histórico da conversa
    messages.extend(chat_history)

    # adiciona pergunta + contexto
    messages.append({
        "role": "user",
        "content": f"Contexto: {context}\n\nPergunta: {question}"
    })

    response = ollama.chat(
        model=model,
        messages=messages
    )

    final_answer = response["message"]["content"]

    # guarda histórico da conversa
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": final_answer})

    return final_answer

# =========================================================
# 9 - LOOP PRINCIPAL DO CHAT RAG
# =========================================================

print("Chat RAG iniciado.")
print("Escreve 'sair' para terminar.\n")

while True:

    # input do utilizador
    user_input = input("Tu: ")

    # condição de saída
    if user_input.lower() == "sair":
        print("Chat terminado.")
        break

    # reescreve pergunta com contexto
    standalone_question = rewrite_question_with_history(user_input, "mistral")

    # gera resposta hipotética (HYDE)
    hypotetical_answer = augment_query_generation(standalone_question, model="mistral")

    # combina pergunta + resposta hipotética
    join_query = f"{standalone_question} - {hypotetical_answer}"

    # pesquisa documentos relevantes
    results = chroma_query(join_query)

    retrieved_docs = results['documents'][0]

    # gera resposta final baseada no contexto recuperado
    final_answer = generate_final_answer(standalone_question, retrieved_docs, model="mistral")

    # imprime resposta
    print(f"Assistente: {final_answer}\n")