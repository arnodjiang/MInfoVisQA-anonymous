pass
import base64
import os
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from dotenv import dotenv_values
MODEL = 'google/gemini-3.8-flash'
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer'], 'additionalProperties': False}

def load_config(root):
    env = dotenv_values(Path(root) / '.env', interpolate=False)
    result = {name: os.environ.get(name) or env.get(name) for name in ('TOKEN_ROUTER_API_KEY', 'TOKEN_ROUTER_BASE_URL')}
    if not all(result.values()):
        raise ValueError('TOKEN_ROUTER_API_KEY and TOKEN_ROUTER_BASE_URL are required')
    return result

def endpoint(base, model=MODEL):
    url = urlsplit(base)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Expected an HTTPS base URL without credentials/query/fragment')
    path = url.path.rstrip('/')
    if path.endswith('/v1beta/models'):
        pass
    elif path.endswith('/v1beta'):
        path += '/models'
    elif path.endswith('/v1'):
        path = path[:-3] + '/v1beta/models'
    elif not path:
        path = '/v1beta/models'
    else:
        raise ValueError('Unsupported base path; use origin, /v1, /v1beta, or /v1beta/models')
    if not model or any((c in model for c in ('?', '#', ':', '..'))):
        raise ValueError('Invalid model identifier')
    return urlunsplit((url.scheme, url.netloc, path + '/' + quote(model, safe='/-._') + ':generateContent', '', ''))

def request_body(prompt, question, image_bytes, mime='image/png', temperature=0.7, max_tokens=8192):
    return {'systemInstruction': {'parts': [{'text': prompt}]}, 'contents': [{'role': 'user', 'parts': [{'text': question}, {'inlineData': {'mimeType': mime, 'data': base64.b64encode(image_bytes).decode()}}]}], 'generationConfig': {'temperature': temperature, 'candidateCount': 1, 'maxOutputTokens': max_tokens, 'responseMimeType': 'application/json', 'responseJsonSchema': SCHEMA}}

def answer_text(data):
    candidates = data.get('candidates') or []
    if len(candidates) != 1:
        raise ValueError('Expected exactly one candidate; response may be blocked')
    candidate = candidates[0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError('Non-success finishReason: ' + str(candidate.get('finishReason')))
    parts = candidate.get('content', {}).get('parts', [])
    return ''.join((p['text'] for p in parts if isinstance(p.get('text'), str) and (not p.get('thought'))))

def normalize_usage(meta):
    if not isinstance(meta, dict) or not meta:
        return None
    prompt = meta.get('promptTokenCount')
    total = meta.get('totalTokenCount')
    candidate = meta.get('candidatesTokenCount')
    thought = meta.get('thoughtsTokenCount')
    output = None
    if isinstance(candidate, int) and isinstance(thought, int):
        output = candidate + thought
    elif isinstance(total, int) and isinstance(prompt, int) and (total >= prompt) and (not meta.get('toolUsePromptTokenCount')):
        output = total - prompt
    return {'input_tokens': prompt, 'output_tokens': output, 'total_tokens': total, 'input_tokens_details': {'cached_tokens': meta.get('cachedContentTokenCount')}, 'output_tokens_details': {'reasoning_tokens': thought}, 'visible_candidate_tokens': candidate, 'output_count_definition': 'candidates + thoughts, or total - prompt for this no-tools request'}

def quota_exhausted(payload):
    pass
    if not isinstance(payload, dict):
        return False
    error = payload.get('error')
    objects = [payload] + ([error] if isinstance(error, dict) else [])
    codes = {str(obj.get('code', '')).lower() for obj in objects}
    if codes & {'insufficient_user_quota', 'insufficient_quota', 'insufficient_balance', 'account_quota_exhausted', 'credit_balance_exhausted', 'insufficient_credits'}:
        return True
    messages = ' '.join((str(obj.get('message', '')) for obj in objects)).lower()
    if isinstance(error, str):
        messages += ' ' + error.lower()
    return any((s in messages for s in ('credit limit is insufficient', 'insufficient credit', 'insufficient balance', 'credit balance is too low', 'quota is running low', 'quota has been exhausted', '余额不足', '额度不足', '余额已耗尽', '额度已用尽')))
