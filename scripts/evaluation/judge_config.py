pass
import os
from scripts.openai_config import load

def resolve(config, model=None):
    config = dict(config)
    for suffix in ('MODEL', 'BASE_URL', 'API_KEY'):
        name = 'JUDGE_' + suffix
        value = os.environ.get(name) or config.get(name)
        if value:
            config['OPENAI_' + suffix] = value
    if model:
        config['OPENAI_MODEL'] = model
    return config

def load_judge(root, model=None):
    return resolve(load(root), model)
