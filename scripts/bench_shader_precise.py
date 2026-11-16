#!/usr/bin/env python3
"""Build and compare an isolated native shader-rewrite experiment, without Metal.

Synthetic inputs are generated here; --sources may point at local .metal files.
Only measurements and source hashes are written, never shader text or paths.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'dm/gfx/experiments/native_precise'
sys.path.insert(0, str(ROOT / 'dm/gfx/tools'))
sys.path.insert(0, str(EXPERIMENT))
from dmsh_rewrite import precise as upstream  # noqa: E402
prefix_python = importlib.import_module('candidates').prefix_python


def build_native():
    cxx = shlex.split(os.environ.get('CXX') or sysconfig.get_config_var('CXX') or 'c++')
    if not os.environ.get('CXX') and not shutil.which(cxx[0]): cxx = ['c++']
    output = EXPERIMENT / ('_precise_native' + sysconfig.get_config_var('EXT_SUFFIX'))
    command = cxx + ['-O3', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                     '-I' + sysconfig.get_path('include'), str(EXPERIMENT / 'precise_native.cpp'), '-o', str(output)]
    if sys.platform == 'darwin': command += ['-undefined', 'dynamic_lookup']
    subprocess.run(command, check=True)
    version = subprocess.run(cxx + ['--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return version


def synthetic_sources():
    # Sizes bracket the 220 KB average reported in upstream issue #1036.
    plain = '    float _17 = fma(_12, _13, _14);\n'
    matched = '    float _18 = fast::clamp(fast::min(_12, _13), fast::max(_14, 0.0), 1.0);\n'
    decoy = '    float _19 = myfast::min(_12, _13); // fast::minimum( fast::min (\n'
    for size in (1024, 32 * 1024, 220 * 1024, 1024 * 1024):
        for name, unit in (('absent', plain), ('sparse', matched + plain * 127),
                           ('dense', matched), ('decoys', decoy)):
            text = (unit * (size // len(unit) + 1))[:size]
            yield f'synthetic/{size}/{name}', text


def fixture_sources():
    """Use the repository's existing synthetic regression fixture, not game code."""
    path = ROOT / 'dm/gfx/tests/test_dmsh_bit0_mask.py'
    spec = importlib.util.spec_from_file_location('_precise_fixture', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for count in (4, 3200):
        source = module.reduced(count)
        yield f'upstream-bit0-fixture/{count}/original', source
        # Labeled derivative, so this is not mistaken for measured game shader density.
        yield f'upstream-bit0-fixture/{count}/with-calls', source.replace('fma(', 'fast::clamp(')


def time_batch(function, source, loops):
    start = time.perf_counter_ns()
    for _ in range(loops): function(source)
    return time.perf_counter_ns() - start


def compare(sources, functions, repeats, target_ms):
    results, rng = [], random.Random(1932)
    for label, source in sources:
        expected = upstream(source)
        for name, function in functions.items():
            if function(source) != expected: raise AssertionError(f'output differs: {label}: {name}')
        loops, samples = {}, {name: [] for name in functions}
        for name, function in functions.items():
            function(source)  # warm imports, regex cache and Unicode representation
            n = 1
            while True:
                elapsed = time_batch(function, source, n)
                if elapsed >= target_ms * 1_000_000 or n >= 1 << 20: break
                n *= 2
            loops[name] = n
        for _ in range(repeats):
            order = list(functions)
            rng.shuffle(order)  # interleave implementations instead of timing all old, then all new
            for name in order:
                samples[name].append(time_batch(functions[name], source, loops[name]) / loops[name] / 1000)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        row = {'case': label, 'characters': len(source), 'replacements': (len(expected) - len(source)) // 3,
               'source_sha256': hashlib.sha256(source.encode('utf-8', errors='surrogatepass')).hexdigest(),
               'output_equal': True, 'microseconds_per_call': medians,
               'samples_microseconds': samples, 'iterations_per_sample': loops,
               'speedup_vs_upstream': {name: medians['upstream'] / value for name, value in medians.items()}}
        results.append(row)
        print(f"{label}: upstream {medians['upstream']:.1f} us; "
              f"Python {row['speedup_vs_upstream']['prefix_python']:.2f}x; "
              f"C++ {row['speedup_vs_upstream']['native']:.2f}x", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='compile the C++ extension using this Python')
    parser.add_argument('--output', type=Path, help='save measurements as JSON')
    parser.add_argument('--sources', type=Path, help='local directory of .metal sources; never copied into results')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=15)
    args = parser.parse_args()
    if args.repeats < 3 or args.target_ms <= 0 or args.limit < 1:
        parser.error('repeats >= 3, target-ms > 0 and limit >= 1 are required')
    compiler = build_native() if args.build else 'existing extension; compiler not recorded'
    importlib.invalidate_caches()
    native = importlib.import_module('_precise_native').precise
    functions = {'upstream': upstream, 'prefix_python': prefix_python, 'native': native}
    if args.sources:
        paths = sorted(args.sources.rglob('*.metal'))[:args.limit]
        if not paths: parser.error('no .metal files under --sources')
        sources = [(f'local/{i}', path.read_text(encoding='utf-8')) for i, path in enumerate(paths)]
    else:
        sources = list(synthetic_sources()) + list(fixture_sources())
    baseline = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    result = {'scope': 'single source-text rewrite only; not whole shader compilation',
              'input_type': 'local .metal files' if args.sources else 'synthetic, no game shader files',
              'git_revision': baseline, 'python': sys.version, 'platform': sys.platform,
              'architecture': platform.machine(), 'compiler': compiler,
              'upstream_module_sha256': hashlib.sha256((ROOT / 'dm/gfx/tools/dmsh_rewrite.py').read_bytes()).hexdigest(),
              'candidate_sha256': {name: hashlib.sha256((EXPERIMENT / name).read_bytes()).hexdigest()
                                   for name in ('precise_native.cpp', 'candidates.py')},
              'repeats': args.repeats, 'results': compare(sources, functions, args.repeats, args.target_ms)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
