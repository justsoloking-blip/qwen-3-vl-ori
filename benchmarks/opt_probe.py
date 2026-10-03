#!/usr/bin/env python3
"""Neutral streaming benchmark and server-side torch.profiler client."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath
import statistics
import time
import urllib.request
import uuid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('runtime/baseline'))
    parser.add_argument('--api-url', help='Server root or /v1 URL; defaults to runtime/config.json')
    parser.add_argument('--model', help='Served model ID; defaults to runtime/config.json')
    parser.add_argument('--api-key-file', type=Path, help='Defaults to runtime/api.key; MODEL_API_KEY takes precedence')
    parser.add_argument('--out', type=Path, help='Client result directory; defaults to runtime/results')
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--cases', default='short,long,image,concurrent')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--profile-tokens', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--server-profile-dir', help='Absolute directory ON THE SERVER; required with --profile')
    args = parser.parse_args()
    args.cases = [case.strip() for case in args.cases.split(',') if case.strip()]
    if not args.cases or any(case not in {'short', 'long', 'image', 'concurrent'} for case in args.cases):
        parser.error('--cases must contain short,long,image,concurrent')
    if min(args.rounds, args.concurrency) < 1 or min(args.output_tokens, args.profile_tokens) < 2 or args.timeout <= 0:
        parser.error('rounds/concurrency/timeout must be positive; output/profile tokens must be >= 2')
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        parser.error('--label must use letters, digits, hyphens or underscores')
    if args.profile and (not args.server_profile_dir or not args.server_profile_dir.startswith('/')):
        parser.error('--profile requires an absolute --server-profile-dir on the inference server')
    return args


def connection(args):
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
        raise ValueError('Prepare runtime/config.json first, or provide both --api-url and --model.')
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        key_path = args.api_key_file or args.runtime / 'api.key'
        if not key_path.is_file():
            raise ValueError('No API key: prepare the runtime, set MODEL_API_KEY, or use --api-key-file.')
        key = key_path.read_text().strip()
    if not key:
        raise ValueError('The API key is empty.')
    return url, model, key


def messages(case):
    nonce = uuid.uuid4().hex
    if case == 'image':
        from PIL import Image, ImageDraw
        image = Image.new('RGB', (1024, 1024), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((64, 64, 480, 480), fill='red')
        draw.ellipse((520, 520, 960, 960), fill='blue')
        draw.text((20, 980), nonce, fill='black')
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
        content = [
            {'type': 'text', 'text': f'Trial {nonce}. Describe the image and geometric relationships in detail.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}},
        ]
    else:
        content = f'Trial {nonce}.\n' + ('这是一份关于计算系统的背景材料。请分析计算、内存访问、通信以及请求调度。\n' * (160 if case == 'long' else 20)) + '\n请详细解释，持续回答。'
    return [{'role': 'user', 'content': content}]


def main():
    args = parse_args()
    url, model, key = connection(args)
    out = args.out or args.runtime / 'results'
    out.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, data):
        request = urllib.request.Request(url + path, data=json.dumps(data).encode(), headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key,
        })
        return opener.open(request, timeout=args.timeout)

    def request(case, tokens):
        payload = {'model': model, 'messages': messages(case), 'temperature': 0,
                   'max_tokens': tokens, 'ignore_eos': True, 'stream': True,
                   'stream_options': {'include_usage': True}}
        times, usage, done = [], None, False
        started = time.perf_counter()
        with api('/v1/chat/completions', payload) as response:
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
                if choices and choices[0].get('delta', {}).get('content'):
                    times.append(time.perf_counter())
                if event.get('usage'):
                    usage = event['usage']
        finished = time.perf_counter()
        if not done:
            raise RuntimeError(f'{case}: SSE ended without [DONE]; check the server/connection.')
        if not times:
            raise RuntimeError(f'{case}: no nonempty text SSE chunk; TTFT cannot be measured.')
        if len(times) == 1:
            raise RuntimeError(f'{case}: only one text SSE chunk; inspect buffering/streaming before measuring decode performance.')
        if not usage or usage.get('completion_tokens') != tokens:
            actual = None if not usage else usage.get('completion_tokens')
            raise RuntimeError(f'{case}: expected fixed output {tokens} tokens, got {actual}; this probe requires SGLang ignore_eos and usage reporting.')
        return {'case': case, 'elapsed_s': finished - started, 'ttft_ms': 1000 * (times[0] - started),
                'tpot_ms': 1000 * (finished - times[0]) / (tokens - 1),
                'text_chunks': len(times), 'chunk_gap_p50_ms': statistics.median(1000 * (b - a) for a, b in zip(times, times[1:])),
                'usage': usage}

    def batch(case, tokens):
        concurrency = args.concurrency if case == 'concurrent' else 1
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: request('short' if case == 'concurrent' else case, tokens), range(concurrency)))
        wall = time.perf_counter() - started
        return {'case': case, 'concurrency': concurrency, 'wall_s': wall,
                'output_tokens_per_s': sum(item['usage']['completion_tokens'] for item in results) / wall,
                'requests': results}

    report = {'label': args.label, 'model': model, 'api_url': url, 'profiled': args.profile,
              'started_at': datetime.now(timezone.utc).isoformat(), 'results': [],
              'metrics': {'ttft': 'HTTP start to first nonempty text SSE chunk',
                          'tpot': '(HTTP completion - first nonempty text chunk) / (completion_tokens - 1)',
                          'throughput': 'total output tokens / batch wall time; includes prefill and HTTP overhead'}}
    result_file = out / (args.label + ('-profile' if args.profile else '-baseline') + '.json')
    if result_file.exists():
        raise ValueError(f'{result_file} exists; choose another --label or --out to preserve the previous run.')
    for case in args.cases:
        print('WARMUP', case, flush=True)
        batch(case, 16)
        batch(case, 16)
        if args.profile:
            directory = str(PurePosixPath(args.server_profile_dir) / args.label / case)
            print('SERVER_PROFILE_DIR', directory, flush=True)
            with api('/start_profile', {'output_dir': directory, 'activities': ['CPU', 'GPU'],
                                      'with_stack': False, 'record_shapes': True, 'profile_prefix': case}) as response:
                response.read()
            try:
                result = batch(case, args.profile_tokens)
                result['server_trace_directory'] = directory
                report['results'].append(result)
            finally:
                with api('/stop_profile', {}) as response:
                    response.read()
        else:
            for repeat in range(args.rounds):
                result = batch(case, args.output_tokens)
                result['repeat'] = repeat
                report['results'].append(result)
                print(json.dumps(result), flush=True)
        result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved', result_file, flush=True)


if __name__ == '__main__':
    main()
