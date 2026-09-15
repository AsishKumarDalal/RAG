import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

k = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
assert k and len(k) > 10, "Missing GOOGLE_API_KEY in .env"
c = genai.Client(api_key=k)
ms = list(c.models.list())
print(f"TOTAL: {len(ms)}")
for m in sorted(ms, key=lambda x: x.name):
    print(f"- {m.name} | display={getattr(m,'display_name',None)} | in={getattr(m,'input_token_limit',None)} out={getattr(m,'output_token_limit',None)}")
