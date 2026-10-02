pass
import json
from openai import OpenAI

def response_text(response):
    text = getattr(response, 'output_text', None)
    if text:
        return text
    chunks = []
    for item in getattr(response, 'output', []) or []:
        for content in getattr(item, 'content', []) or []:
            value = getattr(content, 'text', None)
            if value:
                chunks.append(value)
    return ''.join(chunks)

def create(config, prompt, payload, *, image_uris=(), max_tokens=12000, timeout=180):
    content = [{'type': 'input_text', 'text': json.dumps(payload, ensure_ascii=False)}]
    content.extend(({'type': 'input_image', 'image_url': uri, 'detail': 'high'} for uri in image_uris))
    with OpenAI(api_key=config['OPENAI_API_KEY'], base_url=config.get('OPENAI_BASE_URL') or None, timeout=timeout, max_retries=0) as client:
        return client.responses.create(model=config['OPENAI_MODEL'], instructions=prompt, input=[{'role': 'user', 'content': content}], max_output_tokens=max_tokens)
