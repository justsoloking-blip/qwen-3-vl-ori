#!/usr/bin/env python3
"""Check authenticated model, text, image and streaming APIs; not a quality benchmark."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('runtime/baseline'))
    parser.add_argument('--api-url', help='Server root or /v1 URL; defaults to runtime/config.json')
    parser.add_argument('--model', help='Served model ID; defaults to runtime/config.json')
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--out', type=Path, help='Result JSON path; defaults to runtime/results/smoke.json')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    config_path = args.runtime / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    server_args = config.get('arguments', [])
    model = args.model
    if not model and '--served-model-name' in server_args:
        model = server_args[server_args.index('--served-model-name') + 1]
    url = (args.api_url or config.get('base_url', '')).rstrip('/')
    if url.endswith('/v1'):
        url = url[:-3]
    if not model or not url:
        parser.error('Prepare runtime/config.json first, or provide --api-url and --model.')
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        key_path = args.api_key_file or args.runtime / 'api.key'
        if not key_path.is_file():
            parser.error('No API key: prepare the runtime, set MODEL_API_KEY, or use --api-key-file.')
        key = key_path.read_text().strip()
    if not key:
        parser.error('API key is empty')
    out = args.out or args.runtime / 'results' / 'smoke.json'
    if out.exists():
        parser.error(f'{out} exists; choose --out to preserve the previous result.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, data=None, auth=True):
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer ' + key
        payload = None if data is None else json.dumps(data).encode()
        return opener.open(urllib.request.Request(url + path, data=payload, headers=headers), timeout=args.timeout)

    report = {'model': model, 'api_url': url, 'tested_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'API authentication, availability and response shape; answers are recorded for manual inspection'}
    try:
        with request('/v1/models', auth=False):
            raise RuntimeError('Unauthenticated /v1/models request was accepted.')
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403):
            raise
        report['unauthenticated_status'] = exc.code
    with request('/v1/models') as response:
        models = json.load(response)
    if model not in [item['id'] for item in models.get('data', [])]:
        raise RuntimeError(f'Configured model ID {model!r} is absent from /v1/models.')
    report['models'] = models
    with request('/health') as response:
        if response.status != 200:
            raise RuntimeError('/health did not return 200.')
        report['health_status'] = response.status

    def chat(content):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': content}],
                   'temperature': 0, 'max_tokens': 64}
        with request('/v1/chat/completions', payload) as response:
            result = json.load(response)
        choices = result.get('choices', [])
        answer = choices[0].get('message', {}).get('content') if choices else None
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError('Chat API returned no nonempty text answer.')
        return {'answer': answer, 'usage': result.get('usage')}

    report['text'] = chat('请用一句简短中文介绍自己。')
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (448, 224), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 48, 160, 176), fill='red')
    draw.ellipse((280, 48, 408, 176), fill='blue')
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    report['image'] = chat([{'type': 'image_url', 'image_url': {'url': data_url}},
                            {'type': 'text', 'text': 'Describe the objects from left to right in one short sentence.'}])
    done, chunks, text = False, 0, []
    payload = {'model': model, 'messages': [{'role': 'user', 'content': '请用一句简短中文问候。'}],
               'temperature': 0, 'max_tokens': 64, 'stream': True}
    with request('/v1/chat/completions', payload) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                continue
            if not data:
                continue
            event = json.loads(data)
            choices = event.get('choices', [])
            content = choices[0].get('delta', {}).get('content') if choices else None
            if content:
                text.append(content)
                chunks += 1
    if not done or not ''.join(text).strip():
        raise RuntimeError('Streaming API must return nonempty text and [DONE].')
    report['streaming'] = {'completed': True, 'text_chunks': chunks, 'answer': ''.join(text)}
    report['passed'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('API smoke passed. Answers are recorded; this is not a model-quality evaluation.')
    print('Saved', out)


if __name__ == '__main__':
    main()
