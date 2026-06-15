from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

# 1. Створюємо інструменти
final_answer = FinalAnswerTool()
search_tool = DuckDuckGoSearchTool() # Додаємо пошук
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# 2. Налаштування моделі
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct', # Повертаємо стандартну модель
    custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# 3. ПЕРЕДАЄМО ВСІ ІНСТРУМЕНТИ АГЕНТУ
agent = CodeAgent(
    model=model,
    tools=[
        final_answer, 
        get_current_time_in_timezone, 
        image_generation_tool, 
        search_tool, # Додали пошук у список
        my_custom_tool
    ], 
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
    additional_authorized_imports=['pytz', 'datetime'] 
)

GradioUI(agent).launch()