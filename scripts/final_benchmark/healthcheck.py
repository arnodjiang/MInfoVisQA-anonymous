pass
import json
import os
from urllib.parse import urlsplit
import httpx
from dotenv import dotenv_values
from scripts.openai_config import load as load_openai_config
from .pipeline import ROOT

def main():
    config = load_openai_config(ROOT)
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
        if os.environ.get(key):
            config[key] = os.environ[key]
    proxies = {}
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        if os.environ.get(key):
            u = urlsplit(os.environ[key])
            proxies[key] = {'scheme': u.scheme, 'host': u.hostname, 'port': u.port}
    print(json.dumps({'proxy_endpoints': proxies}), flush=True)
    try:
        with httpx.Client(timeout=20) as client:
            response = client.get(config['OPENAI_BASE_URL'].rstrip('/') + '/models', headers={'Authorization': 'Bearer ' + config['OPENAI_API_KEY']})
        print(json.dumps({'http_status': response.status_code, 'content_type': response.headers.get('content-type'), 'bytes_received': len(response.content)}), flush=True)
    except Exception as exc:
        chain = []
        current = exc
        while current and len(chain) < 6:
            chain.append({'type': type(current).__name__, 'errno': getattr(current, 'errno', None)})
            current = current.__cause__
        print(json.dumps({'error_chain': chain}), flush=True)
        raise SystemExit(1)
if __name__ == '__main__':
    main()
