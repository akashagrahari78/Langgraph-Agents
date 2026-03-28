import os

from google import genai
from google.genai import types
from PIL import Image

api_key = os.getenv("GEMINI_API_KEY")
print(api_key)
if not api_key:
    raise ValueError(
        "Missing GEMINI_API_KEY. In PowerShell run: $env:GEMINI_API_KEY='your_api_key'"
    )

client = genai.Client(api_key=api_key)

prompt = ("Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme")
response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[prompt],
)

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = part.as_image()
        image.save("generated_image.png")