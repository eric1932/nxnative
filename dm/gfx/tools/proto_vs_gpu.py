#!/usr/bin/env python3
# PROTOTYPE, beside proto_vs_shared.py: the GPU compiler's code for each form of a vertex shader's main0, from the libraries `proto_vs_shared.py libs` left under <work>/gpu/<form>/.
# For each form and each of <key>_b and <key>_bp, proto_vs_gpu (proto_vs_gpu.m) adds a pipeline of every shader's entry point to one binary archive: the archive's bytes and the
# time to add them, against form A's. A pipeline alone in an archive is also measured per shader, for the median and the largest change (each archive also holds the fragment
# function's code, the same in every one, so the vertex function's own change is a little larger than the figure).
# Usage: proto_vs_gpu.py <work folder> <proto_vs_gpu binary>
import os, sys, subprocess, statistics as st

FORMS = ('A', 'Bs', 'Bi', 'Cs', 'Ci', 'Ch')
ENTRY = {'b': {f: 'main0' for f in FORMS}, 'bp': {'A': 'main0', 'Bs': 'main0', 'Bi': 'main0', 'Cs': 'main0_p', 'Ci': 'main0_p', 'Ch': 'main0_p'}}


def run(tool, archive, entry, libs):
    r = subprocess.run([tool, archive, entry, *libs], capture_output=True, text=True)
    if r.returncode: sys.exit(f'{tool}: {r.stderr[:400]}')
    added, failed, size, ms = r.stdout.split('\t')
    return int(added), int(failed), int(size), float(ms), r.stderr


def main():
    work, tool = sys.argv[1], sys.argv[2]; gpu = os.path.join(work, 'gpu'); arch = os.path.join(gpu, 'archive.bin')
    keys = sorted(n[:-len('_bp.metallib')] for n in os.listdir(os.path.join(gpu, 'A')) if n.endswith('_bp.metallib'))
    keys = [k for k in keys if all(os.path.exists(os.path.join(gpu, f, f'{k}_{p}.metallib')) for f in FORMS for p in ('b', 'bp'))]
    print(f'{len(keys)} vertex shaders with every form\'s libraries')
    for piece in ('bp', 'b'):
        base = None; per = {}
        for f in FORMS:
            libs = [os.path.join(gpu, f, f'{k}_{piece}.metallib') for k in keys]
            added, failed, size, ms, err = run(tool, arch, ENTRY[piece][f], libs)
            per[f] = [run(tool, arch, ENTRY[piece][f], [p])[2] for p in libs]
            if f == 'A': base = (size, ms)
            ch = [100.0 * (a - b) / b for a, b in zip(per[f], per['A'])]
            print(f'  <key>_{piece} {ENTRY[piece][f]:8} {f:3} {added} pipelines, {failed} failed; one archive {size:9d} bytes ({100.0 * (size - base[0]) / base[0]:+5.1f}%), {ms:7.0f} ms to add ({100.0 * (ms - base[1]) / base[1]:+5.1f}%);'
                  f' alone: median {st.median(ch):+5.1f}%, largest {max(ch):+5.1f}%, smallest {min(ch):+5.1f}%, {sum(a == b for a, b in zip(per[f], per["A"]))} the same size' + (f'\n{err[:300]}' if err else ''))
    os.unlink(arch)


if __name__ == '__main__': main()
