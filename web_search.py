import requests
from urllib.parse import quote

def search_wikipedia(query: str) -> str:
    """
    Recherche la première phrase sur Wikipedia correspondant à la query.
    """
    url = "https://fr.wikipedia.org/w/api.php"
    # Encode correctement la query
    query_encoded = quote(query)
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query_encoded,
        "utf8": 1,
        "format": "json",
        "srlimit": 1
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        # Vérifie que la réponse est bien JSON
        try:
            data = response.json()
        except ValueError:
            return "Error: Wikipedia did not return JSON."

        if "query" in data and data["query"]["search"]:
            snippet = data["query"]["search"][0]["snippet"]
            snippet = snippet.replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            return snippet + "..."
        return "No Wikipedia result."
    except Exception as e:
        return f"Error during Wikipedia search: {e}"
