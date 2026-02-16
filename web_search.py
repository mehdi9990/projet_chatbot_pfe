import requests
from bs4 import BeautifulSoup

def search_web(query: str) -> str:
    try:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        snippets = []
        for g in soup.find_all('div'):
            text = g.get_text().strip()
            if text and len(text) > 50:
                snippets.append(text)

        return "\n\n".join(snippets[:3]) if snippets else "No web result found."

    except:
        return "No web result found."
