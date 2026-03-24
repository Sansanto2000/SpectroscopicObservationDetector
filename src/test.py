from dotenv import load_dotenv
import os

load_dotenv()
print('variable de entorno', os.getenv("RANDOM_SEED"))