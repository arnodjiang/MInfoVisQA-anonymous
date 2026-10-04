pass
import os
from pathlib import Path
from dotenv import dotenv_values
MODEL = 'gpt-6-astra'

def load(root):
    root = Path(root)
    config = dict(dotenv_values(root / '.env', interpolate=False))
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'TRANSLATION_BACKEND', 'GOOGLE_TRANSLATE_API_KEY', 'GOOGLE_TRANSLATE_ENDPOINT'):
        if os.environ.get(key):
            config[key] = os.environ[key]
    configured = os.environ.get('OPENAI_MODEL') or config.get('OPENAI_MODEL')
    config['OPENAI_MODEL'] = configured or MODEL
    return config
