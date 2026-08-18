"""
Агент-дослідник на smolagents: шукає в мережі, відкриває знайдені сторінки і
відповідає з посиланнями, показуючи кожен свій крок.

Це переробка шаблону з курсу HF AI Agents. Від шаблону лишились `prompts.yaml`
і `Gradio_UI.py` (звідти береться лише `pull_messages_from_step` — розбір кроку
агента в повідомлення чату; код апстрімовий і навмисно не форкається).
Усе інше переписане, і ось чому:

- **Space мовчав.** Інференс HF з 401 для анонімних запитів, а секретів у Space
  не було задано жодного. Токен тепер читається з оточення, а якщо його немає —
  інтерфейс каже це прямо, а не мовчить.
- **Агент був спільною пам'яттю на всіх.** `reset_agent_memory=False` на
  публічному Space означає, що діалог одного відвідувача тягнеться в запуск
  наступного: чужий контекст у відповіді і рахунок за токени, який росте сам.
- **Інструмент-заглушка.** `my_custom_tool` повертав "What magic will you
  build?" і при цьому був у списку інструментів агента, тобто модель могла його
  викликати і отримати сміття.
- **`visit_webpage` падав** з NameError (використовував `re` без імпорту) і не
  був підключений — пошук без уміння відкрити знайдене це половина інструмента.
- **Генерація картинок тягнулась через `load_tool(..., trust_remote_code=True)`**,
  тобто виконання коду з чужого репозиторію. Прибрано свідомо.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
from collections import deque

import gradio as gr
import pytz
import yaml
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.memory import ActionStep

from Gradio_UI import pull_messages_from_step
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("agent-space")

try:  # локальний запуск: .env і мережі з TLS-інспекцією
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = (os.environ.get("HF_TOKEN") or "").strip()

# Три обмеження, а не одне. Крок-ліміт не рятує від довгого кроку, таймаут не
# рятує від натовпу, а погодинний ліміт не рятує від одного запиту, що зациклився.
MAX_STEPS = int(os.environ.get("MAX_STEPS", "6"))
RUN_TIMEOUT_S = int(os.environ.get("RUN_TIMEOUT_S", "120"))
RUNS_PER_HOUR = int(os.environ.get("RUNS_PER_HOUR", "20"))

GITHUB_URL = "https://github.com/LeonidShamarin"


# --------------------------------------------------------------------------- #
# Інструменти
# --------------------------------------------------------------------------- #


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.

    Args:
        timezone: A valid IANA timezone name, e.g. 'Europe/Kyiv' or 'America/New_York'.
    """
    try:
        tz = pytz.timezone(timezone)
    except Exception:
        # Помилку варто повертати корисною: модель з підказкою виправляється на
        # наступному кроці, а з голим "Error" просто пробує те саме ще раз.
        needle = timezone.strip().lower().replace(" ", "_")
        close = [z for z in pytz.all_timezones if needle in z.lower()][:8]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/Kyiv'.{hint}"

    now = datetime.datetime.now(tz)
    return f"The current local time in {timezone} is {now:%Y-%m-%d %H:%M:%S} ({now:%Z, UTC%z})."


class ResilientSearchTool(DuckDuckGoSearchTool):
    """
    DuckDuckGo регулярно віддає rate limit, і базовий інструмент на це кидає
    виняток — а виняток усередині циклу агента вбиває весь запуск. Тут відмова
    повертається текстом: агент бачить причину і може переформулювати запит.
    """

    def forward(self, query: str) -> str:
        try:
            return super().forward(query)
        except Exception as exc:  # noqa: BLE001 — навмисно віддаємо агенту, а не нагору
            logger.warning("Пошук не вдався: %s", exc)
            return (
                f"Search failed for '{query}': {str(exc)[:200]}. "
                "The search backend may be rate-limiting. Try a shorter, different query, "
                "or answer from what you already know and say the search was unavailable."
            )


# --------------------------------------------------------------------------- #
# Агент
# --------------------------------------------------------------------------- #

_agent = None
_agent_lock = threading.Lock()
# Агент один на процес і має внутрішній стан, тому паралельні запуски треба
# серіалізувати — інакше кроки двох відвідувачів перемішуються в одній пам'яті.
_run_lock = threading.Lock()


def build_agent() -> CodeAgent:
    with open("prompts.yaml", "r", encoding="utf-8") as stream:
        prompt_templates = yaml.safe_load(stream)

    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id=MODEL_ID,
        custom_role_conversions=None,
    )
    return CodeAgent(
        model=model,
        tools=[
            FinalAnswerTool(),
            ResilientSearchTool(),
            VisitWebpageTool(),
            get_current_time_in_timezone,
        ],
        max_steps=MAX_STEPS,
        verbosity_level=1,
        prompt_templates=prompt_templates,
        additional_authorized_imports=["pytz", "datetime"],
    )


def get_agent() -> CodeAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = build_agent()
                logger.info("Агент зібраний: модель %s, крок-ліміт %s", MODEL_ID, MAX_STEPS)
    return _agent


# --------------------------------------------------------------------------- #
# Бюджет запусків
# --------------------------------------------------------------------------- #

_runs: deque[float] = deque()
_runs_lock = threading.Lock()


def take_run_slot() -> bool:
    """Ковзне вікно в годину: Space публічний, а токен інференсу особистий."""
    now = time.monotonic()
    with _runs_lock:
        while _runs and now - _runs[0] > 3600:
            _runs.popleft()
        if len(_runs) >= RUNS_PER_HOUR:
            return False
        _runs.append(now)
        return True


# --------------------------------------------------------------------------- #
# Інтерфейс
# --------------------------------------------------------------------------- #


def _note(text: str) -> gr.ChatMessage:
    return gr.ChatMessage(role="assistant", content=text)


def stream_agent(agent: CodeAgent, task: str):
    """
    Свій потік замість `stream_to_gradio` з апстріму.

    Причина конкретна: та функція вважає останній елемент потоку самою
    відповіддю, а smolagents 1.13 віддає обгортку `FinalAnswerStep` — і в чат
    замість відповіді потрапляв її `repr`, тобто рядок виду
    `FinalAnswerStep(final_answer='...')`. Розбір кроків (`pull_messages_from_step`)
    лишається апстрімовим, переписаний тільки хвіст.
    """
    last = None
    for step in agent.run(task, stream=True, reset=True):
        last = step
        # Лічильники токенів проставляє виклик, а не сам крок — без цього
        # підпис під кроком показує None замість цифр.
        if isinstance(step, ActionStep) and hasattr(agent.model, "last_input_token_count"):
            step.input_token_count = agent.model.last_input_token_count
            step.output_token_count = agent.model.last_output_token_count
        for message in pull_messages_from_step(step):
            yield message

    final = handle_agent_output_types(getattr(last, "final_answer", last))
    if isinstance(final, AgentImage):
        yield gr.ChatMessage(
            role="assistant", content={"path": final.to_string(), "mime_type": "image/png"}
        )
    elif isinstance(final, AgentAudio):
        yield gr.ChatMessage(
            role="assistant", content={"path": final.to_string(), "mime_type": "audio/wav"}
        )
    else:
        text = final.to_string() if isinstance(final, AgentText) else str(final)
        yield gr.ChatMessage(role="assistant", content=f"**Відповідь:**\n\n{text}")


def interact(prompt: str, messages: list):
    messages = list(messages or [])
    prompt = (prompt or "").strip()
    if len(prompt) < 2:
        yield messages, ""
        return

    messages.append(gr.ChatMessage(role="user", content=prompt))
    yield messages, ""

    if not HF_TOKEN:
        messages.append(
            _note(
                "**Немає токена інференсу.** Hugging Face віддає 401 на анонімні запити до "
                "моделі, тому агент без токена не може зробити жодного кроку. Токен задається "
                "у Settings → Variables and secrets як `HF_TOKEN`."
            )
        )
        yield messages, ""
        return

    if not take_run_slot():
        messages.append(
            _note(
                f"**Ліміт демо вичерпано** — {RUNS_PER_HOUR} запусків на годину. Space публічний, "
                "а токен інференсу особистий, тому бюджет обмежений. Спробуйте за трохи."
            )
        )
        yield messages, ""
        return

    if not _run_lock.acquire(blocking=False):
        messages.append(_note("**Агент зараз зайнятий іншим запитом.** Зачекайте кілька секунд."))
        yield messages, ""
        return

    try:
        deadline = time.monotonic() + RUN_TIMEOUT_S
        agent = get_agent()
        # stream_agent запускає агента з reset=True: інакше пам'ять одного
        # відвідувача потрапляє в запуск наступного — і в контекст, і в рахунок.
        for message in stream_agent(agent, prompt):
            messages.append(message)
            yield messages, ""
            if time.monotonic() > deadline:
                messages.append(
                    _note(
                        f"**Зупинено за часом** — запуск перевищив {RUN_TIMEOUT_S} с. "
                        "Вище видно, докуди агент дійшов."
                    )
                )
                yield messages, ""
                return
    except (RecursionError, MemoryError):
        raise
    except Exception as exc:  # noqa: BLE001 — назовні причина, не трейсбек
        logger.exception("Запуск агента впав")
        messages.append(_note(f"**Запуск не вдався.**\n\n`{' '.join(str(exc).split())[:300]}`"))
        yield messages, ""
    finally:
        _run_lock.release()


INTRO = f"""
# Агент-дослідник

Не чат-бот: це **агент** на [smolagents](https://github.com/huggingface/smolagents).
Він пише код, викликає інструменти, читає результат і вирішує, що робити далі —
і кожен цей крок видно в чаті нижче, разом із витраченими токенами.

**Інструменти:** пошук у вебі, відкриття знайденої сторінки (HTML → markdown),
поточний час у будь-якій таймзоні. Модель — `{MODEL_ID}`.

**Межі задані навмисно:** не більше {MAX_STEPS} кроків на задачу, {RUN_TIMEOUT_S} с
на запуск і {RUNS_PER_HOUR} запусків на годину. Агент без обмежень — це інцидент,
а не демо: він може зациклитись і працювати, доки не скінчаться гроші.

Питати можна українською або англійською.
"""

EXAMPLES = [
    "Знайди, що нового вийшло у smolagents цього року, і дай посилання на джерело.",
    "Котра зараз година в Токіо і на скільки це відрізняється від Києва?",
    "Хто зараз тренує збірну України з футболу? Наведи джерело.",
    "Відкрий https://huggingface.co/docs/smolagents/index і скажи, чим CodeAgent відрізняється від ToolCallingAgent.",
]

with gr.Blocks(title="Агент-дослідник", fill_height=True) as demo:
    gr.Markdown(INTRO)

    # У Gradio 6 формат повідомлень єдиний, тому `type="messages"` прибрано —
    # у 6.x цей аргумент уже помилка, а не сумісність.
    chatbot = gr.Chatbot(
        label="Кроки агента",
        height=520,
        resizable=True,
        placeholder="Тут з'явиться хід міркувань: думка → код → результат інструмента.",
    )
    with gr.Row():
        text_input = gr.Textbox(
            label="Задача",
            placeholder="Наприклад: знайди, коли вийшов Gradio 5, і дай посилання",
            lines=2,
            scale=4,
        )
        send = gr.Button("Запустити", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=text_input, examples_per_page=4)
    gr.Markdown(
        "<sub>Space безкоштовний і засинає без трафіку — перше звернення після сну довше. "
        f"Пам'ять агента скидається між запусками навмисно. Код автора — [GitHub]({GITHUB_URL}).</sub>"
    )

    send.click(interact, [text_input, chatbot], [chatbot, text_input])
    text_input.submit(interact, [text_input, chatbot], [chatbot, text_input])

if __name__ == "__main__":
    # concurrency_limit=1 — агент один на процес; черга краща за перемішані запуски.
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
