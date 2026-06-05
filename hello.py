from dotenv import load_dotenv
load_dotenv()
from google import genai
client = genai.Client()

response = client.models.generate_content(
       model="gemini-2.5-flash",
       contents="Say hello in one sentence."
   )

print(response.text)