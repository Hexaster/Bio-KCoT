from openai import OpenAI
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get

client = OpenAI(
    api_key=env("OPENAI_API_KEY") or env("JUDGE_API_KEY", required=True),
    base_url=env("OPENAI_BASE_URL", get("api.judge_base_url")),
)
models = client.models.list()
model_names = sorted(model.id for model in models.data)
for model_name in model_names:
    print(model_name)
