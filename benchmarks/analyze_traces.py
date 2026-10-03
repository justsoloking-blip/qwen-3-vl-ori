#!/usr/bin/env python3
"""Summarize Chrome/Kineto traces; GPU activity intervals are not SM utilization."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import statistics


def category(name):
    name = name.lower()
    if any(term in name for term in ('nccl', 'cross_device', 'allreduce', 'all_reduce')):
        return 'communication'
    if any(term in name for term in ('gemm', 'gemv', 'nvjet', 'triton_mm', 'cutlass_', 'splitkreduce')):
        return 'gemm'
    if any(term in name for term in ('flashattn', 'flash_fwd', 'fmha', 'attention')):
        return 'attention'
    if any(term in name for term in ('rmsnorm', 'layernorm', 'layer_norm')):
        return 'normalization'
    if any(term in name for term in ('silu', 'gelu', 'act_and_mul')):
        return 'activation'
    if 'rope' in name:
        return 'position_encoding'
    if any(term in name for term in ('index', 'copy', 'memcpy', 'fill', 'gather', 'scatter', 'catarray', 'elementwise')):
        return 'layout_elementwise'
    return 'unknown'


def interval_union(intervals):
    total, end = 0.0, -float('inf')
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def numeric_event(event):
    return (isinstance(event, dict) and isinstance(event.get('ts'), (int, float))
            and isinstance(event.get('dur'), (int, float)) and event['dur'] > 0)


def window_stats(events, start=None, stop=None):
    if not events:
        return {'window_ms': 0.0, 'gpu_busy_union_ms': 0.0, 'no_gpu_activity_pct': None,
                'kernel_sum_ms': 0.0, 'kernel_groups_ms': {}, 'kernel_groups_count': {}}
    if start is None:
        start = min(event['ts'] for event in events)
    if stop is None:
        stop = max(event['ts'] + event['dur'] for event in events)
    selected = [event for event in events if event['ts'] < stop and event['ts'] + event['dur'] > start]
    intervals = [(max(event['ts'], start), min(event['ts'] + event['dur'], stop)) for event in selected]
    busy = interval_union(intervals)
    totals, counts = defaultdict(float), defaultdict(int)
    for event, (a, b) in zip(selected, intervals):
        if event.get('cat') == 'kernel':
            group = category(event.get('name', ''))
            totals[group] += b - a
            counts[group] += 1
    span = stop - start
    return {'window_start_us': start, 'window_end_us': stop, 'window_ms': span / 1000,
            'gpu_busy_union_ms': busy / 1000, 'no_gpu_activity_pct': 100 * (1 - busy / span) if span > 0 else None,
            'kernel_sum_ms': sum(totals.values()) / 1000,
            'kernel_groups_ms': {key: value / 1000 for key, value in totals.items()}, 'kernel_groups_count': dict(counts)}


def correlation(event):
    args = event.get('args') or {}
    value = args.get('correlation')
    return None if value is None else str(value)


def analyze(path, root, trim_start, trim_end):
    reader = gzip.open if path.suffix == '.gz' else open
    with reader(path, 'rt', encoding='utf-8') as stream:
        document = json.load(stream)
    if isinstance(document, dict) and isinstance(document.get('traceEvents'), list):
        events = document['traceEvents']
    elif isinstance(document, list):
        events = document
    else:
        return None, [], []
    events = [event for event in events if numeric_event(event)]
    gpu = [event for event in events if event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    kernels = [event for event in gpu if event.get('cat') == 'kernel']
    graphs = sorted((event for event in events if event.get('name') == 'cudaGraphLaunch'), key=lambda event: event['ts'])
    grouped = defaultdict(list)
    graph_ids = {correlation(event) for event in graphs} - {None}
    for event in kernels:
        if correlation(event) in graph_ids:
            grouped[correlation(event)].append(event)
    matched = [(graph, grouped[correlation(graph)]) for graph in graphs if grouped.get(correlation(graph))]
    stable = matched
    if len(matched) > trim_start + trim_end + 1:
        stable = matched[trim_start:len(matched) - trim_end if trim_end else None]
    relative = str(path.relative_to(root))
    parts = path.relative_to(root).parts
    rank_match = re.search(r'(?:TP|rank)[-_](\d+)', path.name, re.IGNORECASE)
    result = {'trace': relative, 'label': parts[-3] if len(parts) >= 3 else 'unknown',
              'case': parts[-2] if len(parts) >= 2 else 'unknown',
              'rank': rank_match.group(1) if rank_match else 'unknown',
              'gpu_event_count': len(gpu), 'graph_replays': len(graphs),
              'correlated_graphs': len(matched), 'stable_graphs': len(stable),
              'whole_trace': window_stats(gpu)}
    if stable:
        start = min(event['ts'] for event in stable[0][1])
        stop = max(event['ts'] + event['dur'] for event in stable[-1][1])
        phase = 'stable_decode'
        result['analysis_scope'] = 'CUDA Graph-correlated stable decode window'
        result['decode'] = window_stats(gpu, start, stop)
        result['graph_kernel_count_median'] = statistics.median(len(items) for _, items in stable)
        result['graph_gpu_span_ms_median'] = statistics.median(
            (max(event['ts'] + event['dur'] for event in items) - min(event['ts'] for event in items)) / 1000
            for _, items in stable)
        result['graph_cpu_launch_ms_median'] = statistics.median(graph['dur'] / 1000 for graph, _ in stable)
        starts = [min(event['ts'] for event in items) for _, items in stable]
        result['graph_start_interval_ms_median'] = statistics.median((b - a) / 1000 for a, b in zip(starts, starts[1:])) if len(starts) > 1 else None
    else:
        phase = 'whole_trace'
        result['analysis_scope'] = 'Whole trace; stable decode cannot be inferred without graph/kernel correlation'
        result['decode'] = None
        start = min((event['ts'] for event in gpu), default=0)
        stop = max((event['ts'] + event['dur'] for event in gpu), default=0)
    totals = defaultdict(list)
    for event in kernels:
        if event['ts'] < stop and event['ts'] + event['dur'] > start:
            totals[event.get('name', '<unnamed>')].append(min(event['ts'] + event['dur'], stop) - max(event['ts'], start))
    rows = [{'trace': relative, 'label': result['label'], 'case': result['case'], 'rank': result['rank'],
             'phase': phase, 'category': category(name), 'kernel': name, 'count': len(durations),
             'total_ms': sum(durations) / 1000, 'mean_us': statistics.mean(durations)} for name, durations in totals.items()]
    result['top_kernels'] = sorted(rows, key=lambda row: row['total_ms'], reverse=True)[:25]
    copies = [{'trace': relative, 'name': event.get('name', '<unnamed>'), 'duration_us': event['dur'],
               'bytes': (event.get('args') or {}).get('bytes')} for event in gpu if event.get('cat') == 'gpu_memcpy']
    return result, rows, copies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runtime/baseline/results'), help='Trace root, or a directory containing traces/')
    parser.add_argument('--out', type=Path, help='Defaults to root/analysis')
    parser.add_argument('--trim-start', type=int, default=5, help='Skip initial correlated graph launches when enough launches exist')
    parser.add_argument('--trim-end', type=int, default=2)
    args = parser.parse_args()
    if min(args.trim_start, args.trim_end) < 0:
        parser.error('trim values must be nonnegative')
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f'trace root does not exist: {root}')
    out = (args.out or root / 'analysis').resolve()
    scan = root / 'traces' if (root / 'traces').is_dir() else root
    paths = sorted(path for path in scan.rglob('*') if path.is_file()
                   and (path.name.endswith('.json') or path.name.endswith('.json.gz'))
                   and not path.is_relative_to(out))
    results, rows, copies, errors = [], [], [], []
    for path in paths:
        try:
            result, new_rows, new_copies = analyze(path, root, args.trim_start, args.trim_end)
        except (OSError, ValueError, TypeError) as exc:
            errors.append({'file': str(path.relative_to(root)), 'error': str(exc)})
            continue
        if result is None:
            continue
        results.append(result)
        rows.extend(new_rows)
        copies.extend(new_copies)
        scope = result['decode'] or result['whole_trace']
        print(result['trace'], 'scope=', result['analysis_scope'], 'window_ms=', round(scope['window_ms'], 3),
              'busy_union_ms=', round(scope['gpu_busy_union_ms'], 3), 'no_gpu_activity_pct=', scope['no_gpu_activity_pct'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'trace-summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    (out / 'memcpy-summary.json').write_text(json.dumps(copies, ensure_ascii=False, indent=2) + '\n')
    notes = {'trace_count': len(results), 'parse_errors': errors,
             'duration_unit': 'Chrome trace timestamps/durations are assumed to be microseconds',
             'scope': 'Per trace file; GPU activity window is not the entire HTTP request',
             'kernel_groups': 'Name-based heuristic; unrecognized kernels stay unknown',
             'activity': 'Interval union of kernel/memcpy/memset events; complement is not SM utilization',
             'kernel_sum': 'Overlapping kernel durations are summed, not critical-path latency',
             'graphs': 'Stable decode requires explicit cudaGraphLaunch/kernel correlation; otherwise report whole trace'}
    (out / 'analysis-notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n')
    fields = ['trace', 'label', 'case', 'rank', 'phase', 'category', 'kernel', 'count', 'total_ms', 'mean_us']
    with (out / 'kernels.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(results)} trace summaries to {out}; parse errors: {len(errors)}')
    if not results:
        raise SystemExit('No Chrome/Kineto trace documents found; fetch server trace files and check --root.')
    if errors:
        raise SystemExit('Some trace files could not be parsed; inspect analysis-notes.json before drawing conclusions.')


if __name__ == '__main__':
    main()
