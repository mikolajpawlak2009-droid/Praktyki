import json
from typing import Dict, List

import requests
from flask import Flask, jsonify, request, abort, Response
try:
    from anthropic import AnthropicFoundry
except ImportError:
    AnthropicFoundry = None
    # Poprzednia wersja polegała na lokalnej instancji openai.ChatCompletion;
    # teraz skupiamy się wyłącznie na kliencie Anthropic, więc brak SDK oznacza
    # jedynie przejście na prosty requests, a nie uruchamianie OpenAI.

# ------------------------------------------------------------
# TUTAJ WPISZ SWOJE KLUCZE
# ------------------------------------------------------------
ANTHROPIC_API_KEY = ""
ANTHROPIC_BASE_URL = "https://llm-praktyki.services.ai.azure.com/anthropic/"
ANTHROPIC_MESSAGES_ENDPOINT = f"{ANTHROPIC_BASE_URL}v1/messages"
ANTHROPIC_PROJECT_ENDPOINT = "https://llm-praktyki.services.ai.azure.com/api/projects/proj-default"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
# W poprzednim pliku wstawiono tutaj numer wersji modelu (20250929) i Azure
# zwracał 400 "invalid anthropic-version". Musi tu być numer API, np. 2023-06-01.
ANTHROPIC_VERSION = "2023-06-01"
ALLOW_MOCKS = False  # True = włączone mocki, False = wyłączone

# Najpierw próbujemy użyć oficjalnego klienta AnthropicFoundry z Azure,
# bo to obsługuje autoryzację i nagłówki automatycznie. Gdy SDK nie jest
# dostępny, później fallbackujemy do surowego requests.
ANTHROPIC_CLIENT = None
if AnthropicFoundry is not None and ANTHROPIC_API_KEY:
    try:
        ANTHROPIC_CLIENT = AnthropicFoundry(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
            default_headers={"anthropic-version": ANTHROPIC_VERSION},
        )
    except Exception as e:
        print("Anthropic client init error:", repr(e))
elif AnthropicFoundry is None:
    print("Anthropic SDK not installed. Run `pip install anthropic` to enable SDK client.")

# ------------------------------------------------------------
# ANTHROPIC AZURE HELPERS
# ------------------------------------------------------------
def _call_anthropic_messages(prompt: str, max_tokens: int = 400) -> str:
    """
    Wywołuje Azure Anthropic Messages API i zwraca odpowiedź tekstową.
    Azure oczekuje nagłówka api-key (nie x-api-key) oraz wersji API.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Brak wartości ANTHROPIC_API_KEY.")

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    texts: List[str] = []

    if ANTHROPIC_CLIENT is not None:
        # W starej wersji było .completions.create(prompt=...), co w Azure Foundry
        # nie działa — trzeba korzystać z Messages API jak poniżej.
        response = ANTHROPIC_CLIENT.messages.create(
            model=ANTHROPIC_MODEL,
            messages=payload["messages"],
            max_tokens=max_tokens,
        )
        for block in getattr(response, "content", []):
            text_value = getattr(block, "text", None)
            if text_value:
                texts.append(text_value)
    else:
        headers = {
            "Content-Type": "application/json",
            "api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION
        }
        response = requests.post(
            ANTHROPIC_MESSAGES_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        content_blocks = data.get("content", [])
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))

    if not texts:
        raise ValueError("Brak tekstu w odpowiedzi Anthropic.")
    return "\n".join(texts).strip()

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def get_holidays(date: str, country_code: str = "PL") -> List[str]:
    year = date.split("-")[0]
    try:
        response = requests.get(f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}", timeout=5)
        response.raise_for_status()
        holidays = response.json()
        return [h.get('localName') or h.get('name') for h in holidays if h.get('date') == date]
    except Exception as e:
        print("get_holidays error:", repr(e))
        return []

def build_prompt(industry: str, date: str, holidays: list):
    holidays_text = ", ".join(holidays) if holidays else "brak"
    prompt = f"""
Jesteś ekspertem od marketingu.
Dziś jest {date}.
Branża: {industry}.
Święta dzisiaj: {holidays_text}.

Podaj 2-3 pomysły na kampanię marketingową w dokładnym formacie JSON, np.:

[
  {{
    "tytuł": "Przykładowy tytuł",
    "opis": "Krótki opis kampanii."
  }}
]

WAŻNE: Zwróć TYLKO czysty JSON bez żadnych komentarzy, wyjaśnień ani znaczników markdown (```json). 
Odpowiedź musi zaczynać się od [ i kończyć się na ].
"""
    return prompt

def _extract_json(text: str):
    if not text:
        return None
    start = None
    for i, ch in enumerate(text):
        if ch in '[{':
            start = i
            break
    if start is None:
        return None
    for j in range(len(text), start, -1):
        candidate = text[start:j]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(candidate, strict=False)
            except Exception:
                continue
    return None


def _strip_code_fence(text: str) -> str:
    """
    Azure Anthropic lubi zwracać treść owiniętą w ```json ... ```.
    Poprzednia wersja kodu próbowała parsować taki fragment wprost i kończyło
    się to ValueError, więc tutaj usuwamy znaczniki przed dalszym przetwarzaniem.
    """
    if not text:
        return text

    raw = text.strip()
    if not raw.startswith("```"):
        return raw

    # Usuwamy pierwszą linię ``` / ```json
    newline_idx = raw.find("\n")
    if newline_idx == -1:
        return raw
    trimmed = raw[newline_idx + 1 :]

    # Szukamy ostatniego wystąpienia ``` i odcinamy je (czasem Azure dodaje tekst
    # po bloku, więc wolimy rfind niż zakładać, że to ostatnia linia).
    closing_idx = trimmed.rfind("```")
    if closing_idx != -1:
        trimmed = trimmed[:closing_idx]

    return trimmed.strip()

# ------------------------------------------------------------
# GENERATE IDEAS
# ------------------------------------------------------------
def generate_ideas(industry: str, date: str, country_code: str = "PL") -> List[Dict]:
    holidays = get_holidays(date, country_code)
    prompt = build_prompt(industry, date, holidays)

    # Anthropic via Azure-hosted Messages API (poprzednio próbowano też OpenAI,
    # ale to środowisko ma działać tylko na Azure Anthropic, stąd uproszczenie)
    if ANTHROPIC_API_KEY:
        try:
            raw = _call_anthropic_messages(prompt)
            text = _strip_code_fence(raw)
            # Próbujemy najpierw bezpośrednio sparsować czysty JSON
            try:
                parsed = json.loads(text)
                if parsed is not None:
                    return parsed
            except json.JSONDecodeError:
                # Jeśli nie zadziałało, próbujemy z _extract_json (backup)
                parsed = _extract_json(text)
                if parsed is not None:
                    return parsed
            raise ValueError(f"Anthropic zwrócił nieparsowalny tekst: {text[:300]}")
        except Exception as e:
            print("Anthropic request failed:", repr(e))

    # 3) Fallback mock
    if ALLOW_MOCKS:
        return [
            {
                "tytuł": f"Szybka promocja dla {industry}",
                "opis": f"Krótkie akcje związane z {', '.join(holidays) if holidays else 'okazją'} na {date}."
            },
            {
                "tytuł": f"Konkurs social dla {industry}",
                "opis": "Konkurs z hashtagiem i nagrodami, promowany w social media."
            }
        ]

    raise RuntimeError("Nie udało się wygenerować pomysłów — Anthropic nie zwrócił poprawnej odpowiedzi.")
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    # Zwracamy surowy tekst 'pong' (bez JSON)
    return Response("pong", mimetype="text/plain")


@app.route("/ideas", methods=["GET"])
def ideas_get():
    # Parametry query: industry, date, country (optional)
    industry = request.args.get("industry")
    date = request.args.get("date")
    country = request.args.get("country", "PL")
    if not industry or not date:
        return jsonify({"error": "Parametry 'industry' i 'date' są wymagane."}), 400
    try:
        ideas = generate_ideas(industry, date, country)
        return jsonify(ideas)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ideas", methods=["POST"])
def ideas_post():
    # Przyjmujemy JSON {"industry": "...", "date": "YYYY-MM-DD", "country": "PL"}
    if not request.is_json:
        return jsonify({"error": "Oczekiwany JSON w body."}), 400
    payload = request.get_json()
    industry = payload.get("industry")
    date = payload.get("date")
    country = payload.get("country", "PL")
    if not industry or not date:
        return jsonify({"error": "'industry' i 'date' są wymagane w JSON."}), 400
    try:
        ideas = generate_ideas(industry, date, country)
        return jsonify(ideas)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Flask wrapper dla generatora pomysłów')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("=== DIAGNOSTYKA ===")
    print("ANTHROPIC_API_KEY:", "TAK" if ANTHROPIC_API_KEY else "NIE")
    print("ANTHROPIC_BASE_URL:", ANTHROPIC_BASE_URL)
    print("ALLOW_MOCKS:", ALLOW_MOCKS)
    print("===================")

    app.run(host=args.host, port=args.port, debug=args.debug)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description='Generuj pomysły marketingowe')
    parser.add_argument('-i', '--industry', help='Branża, np. Cukiernia')
    parser.add_argument('-d', '--date', help='Data (YYYY-MM-DD lub YYYY)')
    parser.add_argument('-c', '--country', default='PL', help='Kod kraju, np. PL')
    args = parser.parse_args()

    industry = args.industry or input('Branża: ').strip()
    date = args.date or input('Data (YYYY-MM-DD lub YYYY): ').strip()
    country = args.country

    if not industry or not date:
        print('Branża i data są wymagane.')
        raise SystemExit(1)

    print("=== DIAGNOSTYKA ===")
    print("ANTHROPIC_API_KEY:", "TAK" if ANTHROPIC_API_KEY else "NIE")
    print("ANTHROPIC_BASE_URL:", ANTHROPIC_BASE_URL)
    print("ANTHROPIC_PROJECT_ENDPOINT:", ANTHROPIC_PROJECT_ENDPOINT)
    print("ALLOW_MOCKS:", ALLOW_MOCKS)
    print("===================")

    try:
        ideas = generate_ideas(industry, date, country)
        print(json.dumps(ideas, ensure_ascii=False, indent=2))
        with open("last_response.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(ideas, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Wystąpił błąd:", str(e))
        tb = traceback.format_exc()
        print(tb[:2000])
        with open("last_response.txt", "w", encoding="utf-8") as f:
            f.write("ERROR:\n")
            f.write(str(e) + "\n\n")
            f.write("TRACEBACK:\n")
            f.write(tb)


