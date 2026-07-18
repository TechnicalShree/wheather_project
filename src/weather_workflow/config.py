import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "openai/gpt-4o-mini"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
