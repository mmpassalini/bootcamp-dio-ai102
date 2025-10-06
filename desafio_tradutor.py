# ===================== Imports e Configurações Globais =====================
import requests
import os
import time
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo 'config.env'
load_dotenv("config.env", override=True)

# --- Configurações da API da Azure ---
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "o4-mini")
AZURE_API_VER = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

# --- Constantes do Script ---
CHAT_API_URL = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VER}"
REQUEST_HEADERS = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
API_TIMEOUT_SECONDS = 180
CHUNK_SIZE_CHARS = 800 # Tamanho de cada bloco de texto para tradução
INTER_REQUEST_DELAY = 0.2 # Pequeno atraso para evitar sobrecarga da API

print("URL da API:", CHAT_API_URL)

# ===================== Funções de Extração e Limpeza =====================

def get_article_content(url: str) -> str | None:
    """Busca o conteúdo de uma URL e extrai o texto principal do artigo."""
    try:
        response = requests.get(url, timeout=40)
        response.raise_for_status() # Lança uma exceção para status de erro (4xx ou 5xx)
    except requests.RequestException as e:
        print(f"Falha ao buscar a URL. Erro: {e}")
        return None

    html_parser = BeautifulSoup(response.text, "html.parser")

    # Remove tags que não contêm conteúdo visível
    for tag in html_parser(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()

    # Tenta encontrar o corpo do artigo usando seletores comuns
    content_selectors = ["div.crayons-article__body", "div#article-body", "article", "main"]
    for selector in content_selectors:
        main_node = html_parser.select_one(selector)
        if main_node and main_node.get_text(strip=True):
            return main_node.get_text(" ", strip=True)

    # Se nenhum seletor funcionar, retorna todo o texto da página como último recurso
    return html_parser.get_text(" ", strip=True)

def clean_markdown_fences(text: str) -> str:
    """Remove as cercas de código (```) do início e do fim de um texto."""
    if not text:
        return ""
    processed_text = text.strip()
    if processed_text.startswith("```"):
        lines = processed_text.splitlines()
        if lines: lines.pop(0) # Remove a primeira linha
        if lines and lines[-1].strip() == "```": lines.pop(-1) # Remove a última linha
        processed_text = "\n".join(lines).strip()
    return processed_text

# ===================== Funções de Interação com a API =====================

def execute_translation_request(text_chunk: str, target_lang: str, max_tokens: int):
    """Monta e executa a chamada para a API de tradução da Azure."""
    system_prompt = (
        "Você é um assistente de tradução de alta qualidade. "
        "Traduza o texto de forma direta e precisa, preservando a formatação Markdown. "
        "Não inclua explicações, comentários ou notas. Responda apenas com o texto traduzido."
    )
    user_prompt = f"Traduza o seguinte texto para o idioma '{target_lang}':\n\n{text_chunk}"

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_completion_tokens": max_tokens,
        "temperature": 0.5, # Temperatura moderada para traduções mais consistentes
    }

    try:
        response = requests.post(CHAT_API_URL, headers=REQUEST_HEADERS, json=payload, timeout=API_TIMEOUT_SECONDS)
        api_response_data = response.json()
    except Exception as e:
        raise SystemExit(f"Erro na requisição ou ao decodificar JSON. Status={response.status_code if 'response' in locals() else 'N/A'}. Detalhes: {e}")

    # Mecanismo de retry inteligente para parâmetros não suportados pela API
    if response.status_code == 400 and isinstance(api_response_data, dict):
        error_info = api_response_data.get("error", {})
        if error_info.get("code") == "unsupported_parameter":
            unsupported_param = error_info.get("param")
            if unsupported_param in payload:
                print(f"Parâmetro '{unsupported_param}' não suportado. Tentando novamente sem ele.")
                payload.pop(unsupported_param, None)
                response = requests.post(CHAT_API_URL, headers=REQUEST_HEADERS, json=payload, timeout=API_TIMEOUT_SECONDS)
                api_response_data = response.json()

    if response.status_code != 200:
        error_details = json.dumps(api_response_data, ensure_ascii=False, indent=2)
        raise SystemExit(f"Erro na API. Status {response.status_code}:\n{error_details}")

    # Extrai os dados relevantes da resposta
    first_choice = (api_response_data.get("choices") or [{}])[0]
    translated_content = (first_choice.get("message") or {}).get("content") or ""
    finish_reason = first_choice.get("finish_reason")
    usage_stats = api_response_data.get("usage", {})

    return clean_markdown_fences(translated_content), finish_reason, usage_stats

def adaptive_translation_handler(text_chunk: str, target_lang: str) -> str:
    """
    Gerencia a tradução de um bloco de texto, adaptando-se a possíveis falhas da API,
    como o limite de tokens, e dividindo o texto se necessário.
    """
    # 1. Tenta traduzir com diferentes limites de tokens, do menor para o maior
    token_limits = [2000, 4000, 6000]
    for tokens in token_limits:
        output, finish_reason, usage = execute_translation_request(text_chunk, target_lang, tokens)
        if output:
            return output  # Sucesso, retorna a tradução
        if finish_reason == "length":
            # A tradução foi interrompida por falta de tokens, tenta com um limite maior
            print(f"[Aviso] A tradução do bloco foi cortada (finish_reason=length). Uso: {usage}")
            continue

    # 2. Se todas as tentativas falharem, divide o bloco e tenta recursivamente
    if len(text_chunk) > 400: # Apenas divide blocos com tamanho razoável
        print("Bloco muito complexo. Dividindo em duas partes...")
        midpoint = len(text_chunk) // 2
        first_half = adaptive_translation_handler(text_chunk[:midpoint], target_lang)
        second_half = adaptive_translation_handler(text_chunk[midpoint:], target_lang)
        return (first_half + "\n\n" + second_half).strip()

    # 3. Última tentativa com um limite de tokens padrão
    print("[Aviso] Tentando uma última vez com tokens padrão.")
    output, _, _ = execute_translation_request(text_chunk, target_lang, 1500)
    return output

# ===================== Função Principal de Orquestração =====================

def translate_full_article(full_text: str, target_lang: str, chunk_size: int) -> str:
    """Divide o texto completo em blocos e orquestra a tradução de cada um."""
    if not full_text:
        return ""

    # Divide o texto em fragmentos do tamanho definido
    text_fragments = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    translated_fragments = []
    total_fragments = len(text_fragments)

    for i, fragment in enumerate(text_fragments, 1):
        print(f"Processando bloco {i}/{total_fragments} (tamanho: {len(fragment)} caracteres)...")
        translated_part = adaptive_translation_handler(fragment, target_lang)
        translated_fragments.append(translated_part or "")
        time.sleep(INTER_REQUEST_DELAY)

    return "\n\n".join(translated_fragments).strip()

# ===================== Execução do Script =====================

def main():
    """Função principal que executa todo o fluxo de tradução."""
    # Teste rápido de funcionalidade
    print("Executando teste de conexão rápida...")
    quick_test_result = adaptive_translation_handler("Hello, world!", "Português do Brasil")
    print(f"Resultado do teste: {quick_test_result}\n")

    # URL do artigo a ser traduzido
    article_url = "[https://dev.to/harshm03/network-layer-internet-protocol-computer-networks-4847](https://dev.to/harshm03/network-layer-internet-protocol-computer-networks-4847)"
    print(f"Iniciando extração de conteúdo da URL: {article_url}")

    original_text = get_article_content(article_url)
    if not original_text:
        print("Não foi possível extrair o texto. Encerrando o programa.")
        return

    print(f"Texto extraído com sucesso ({len(original_text)} caracteres). Iniciando tradução...")
    translated_article_content = translate_full_article(original_text, "Português do Brasil", chunk_size=CHUNK_SIZE_CHARS)

    if not translated_article_content:
        print("A tradução resultou em um texto vazio. Verifique a API e os logs.")
        return
        
    print("\n--- Visualização da Tradução (primeiros 1500 caracteres) ---\n")
    print(translated_article_content[:1500])

    # Salva o resultado em um arquivo Markdown
    output_filename = "artigo_traduzido_versao_modificada.md"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(translated_article_content)
        print(f"\n[SUCESSO] Tradução completa salva no arquivo: {output_filename}")
    except IOError as e:
        print(f"\n[ERRO] Não foi possível salvar o arquivo. Detalhes: {e}")

if __name__ == "__main__":
    main()