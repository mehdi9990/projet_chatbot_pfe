import requests
from bs4 import BeautifulSoup

def search_web(query: str) -> str:
    """
    Effectue une recherche web simple via Bing ou Google (en scrape basique)
    et retourne le texte des premiers résultats.

    ⚠️ Attention : scraping basique, pas une API officielle.
    """
    try:
        # URL de recherche Google (ou Bing) – ici exemple Google
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Récupération des extraits des résultats
        snippets = []
        for g in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            text = g.get_text().strip()
            if text:
                snippets.append(text)

        return "\n\n".join(snippets[:5])  # On limite aux 5 premiers résultats

    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return "No web result found."
