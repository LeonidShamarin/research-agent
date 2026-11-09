from typing import Any, Optional
from smolagents.tools import Tool
import re
import requests
import markdownify

# Вікіпедія та частина інших сайтів віддають 403 на стандартний `python-requests/x.y`:
# їхні правила вимагають, щоб бот назвався і дав спосіб зв'язку. Даємо адресу Space.
HEADERS = {
    "User-Agent": (
        "ResearchAgentDemo/1.0 "
        "(+https://huggingface.co/spaces/LeonidShamarin/First_agent_template) "
        f"python-requests/{requests.__version__}"
    ),
    "Accept-Language": "uk,en;q=0.8",
}


def _main_html(html: str) -> str:
    """Основний вміст сторінки: <main>, інакше <article>, інакше <body>, без
    скриптів, стилів і навігації. Якщо розбір не вдався — сторінка як є."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in node.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    return str(node)


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    inputs = {'url': {'type': 'string', 'description': 'The url of the webpage to visit.'}}
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException

            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20, headers=HEADERS)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Лише основний вміст: на Вікіпедії перші 10 000 символів повної сторінки
            # були меню й навігацією, і до самої статті агент не доходив.
            markdown_content = markdownify(_main_html(response.text)).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return truncate_content(markdown_content, 10000)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
