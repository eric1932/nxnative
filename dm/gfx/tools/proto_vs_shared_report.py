#!/usr/bin/env python3
# PROTOTYPE, beside proto_vs_shared.py: the tables of what it measured. Usage: proto_vs_shared_report.py <work folder> time|check
import os, sys, json, statistics as st

MAIN = ('A', 'Bs', 'Bi', 'Cs', 'Ci', 'Ch', 'Co', 'Cg')
PNAME = {'main0': 'main0_p', 'main0_z': 'main0_zp'}   # the pulling entry points' names in a unit


def cpu(rec, pieces):   # CPU ms of a form's compiles of these pieces (a unit: its one compile and the cut)
    if 'u' in rec['pieces']: return rec['pieces']['u']['metal'] + rec['pieces']['u']['link'] + rec['cut']
    return sum(rec['pieces'][p]['metal'] + rec['pieces'][p]['link'] for p in pieces)


def pct(new, old): return 100.0 * (new - old) / old


def fn(rec, lib, name):   # a function's (bitcode size, digest) in a form's library, under its name there
    f = rec['libs'][lib]['fns']
    return f.get(name) or f.get(PNAME.get(name, name))


def table(rows, forms, pieces, timed, title):
    forms = [f for f in forms if any(f in r['forms'] for r in rows)]   # a run of only some forms (PROTO_FORMS)
    ok = [r for r in rows if all(f in r['forms'] and 'error' not in r['forms'][f] for f in forms)]
    print(f'\n{title}: {len(ok)} shaders that build in every form, of {len(rows)}')
    if not ok: return
    base = forms[0]
    for f in forms:
        fails = sum(1 for r in rows if f in r['forms'] and 'error' in r['forms'][f])
        line = f'  {f:6}'
        if timed:
            t = st.mean(cpu(r['forms'][f], pieces) for r in ok); t0 = st.mean(cpu(r['forms'][base], pieces) for r in ok)
            m = st.mean(sum(p['metal'] for n, p in r['forms'][f]['pieces'].items() if n in pieces or n == 'u') for r in ok)
            lk = st.mean(sum(p['link'] for n, p in r['forms'][f]['pieces'].items() if n in pieces or n == 'u') for r in ok)
            line += f' CPU {t:6.1f} ms a shader (metal {m:6.1f}, metallib {lk:5.1f}), {pct(t, t0):+6.1f}%;'
        line += f' {fails} do not build;'
        for lib, name in (('b', 'main0'), ('b', 'main0_z'), ('b', 'main0_gs'), ('bp', 'main0'), ('bp', 'main0_wl')) + ((('bv', 'main0'),) if 'bv' in pieces else ()):
            ch = [pct(fn(r['forms'][f], lib, name)[0], fn(r['forms'][base], lib, name)[0]) for r in ok if lib in r['forms'][base]['libs'] and fn(r['forms'][base], lib, name) and fn(r['forms'][f], lib, name)]
            same = sum(1 for r in ok if lib in r['forms'][base]['libs'] and fn(r['forms'][base], lib, name) and fn(r['forms'][f], lib, name) == fn(r['forms'][base], lib, name))
            if ch: line += f' {lib}.{name} {st.median(ch):+5.1f}% (max {max(ch):+5.1f}%, {same} same);'
        libs = [pct(sum(r['forms'][f]['libs'][p]['bytes'] for p in pieces), sum(r['forms'][base]['libs'][p]['bytes'] for p in pieces)) for r in ok]
        air = pct(sum(sum(p['air'] for n, p in r['forms'][f]['pieces'].items() if n in pieces or n == 'u') for r in ok), sum(sum(p['air'] for n, p in r['forms'][base]['pieces'].items() if n in pieces) for r in ok))
        line += f' libraries {st.median(libs):+5.1f}% (median), .air {air:+5.1f}% (total)'
        print(line)


def main():
    work, mode = sys.argv[1], sys.argv[2]
    d = json.load(open(os.path.join(work, f'vs_shared_{mode}.json'))); rows = [r for r in d['rows'] if 'forms' in r]; timed = mode.startswith('time')
    print(f'{d["library"]}\n{len(d["rows"])} sampled of {d["with_pull"]} vertex shaders with a pulling library; {len(d["rows"]) - len(rows)} of a form the prototype does not know: '
          + ', '.join(sorted({r["shape"] for r in d["rows"] if "shape" in r})))
    print(f'source {st.mean(r["bytes"] for r in rows) / 1024:.0f} KB on average; wide-line entry points in {sum(r["wl"] for r in rows)}, integer inputs (<key>_bv) in {sum(r["vin"] for r in rows)}, '
          f'other entry points in {sum(bool(r["other"]) for r in rows)}')
    spread = int(sys.argv[3]) if len(sys.argv) > 3 else 40   # the first rows are the evenly spread ones, the rest the largest sources (proto_vs_shared.pick)
    even = [r for r in d['rows'][:spread] if 'forms' in r]; big = [r for r in d['rows'][spread:] if 'forms' in r]
    table(even, MAIN, ('b', 'bp'), timed, f'<key>_b and <key>_bp, the {len(even)} evenly spread ({st.mean(r["bytes"] for r in even) / 1024:.0f} KB of source on average)')
    if big: table(big, MAIN, ('b', 'bp'), timed, f'<key>_b and <key>_bp, the {len(big)} largest sources ({st.mean(r["bytes"] for r in big) / 1024:.0f} KB on average)')
    rows_all = rows; rows = even
    vin = [r for r in rows_all if r['vin']]
    if vin: table(vin, ('A', 'Bs', 'Bi', 'Cs+v', 'Ci+v', 'Ch+v', 'Co+v', 'Cg+v'), ('b', 'bp', 'bv'), timed, '<key>_b, <key>_bp and <key>_bv (shaders with integer inputs)')
    if timed:   # per piece, for the projection: what today's two compiles cost (the evenly spread shaders)
        ok = [r for r in rows if all('error' not in r['forms'][f] for f in MAIN if f in r['forms'])]
        for p in ('b', 'bp'):
            print(f'  today {p}: metal {st.mean(r["forms"]["A"]["pieces"][p]["metal"] for r in ok):6.1f} ms, metallib {st.mean(r["forms"]["A"]["pieces"][p]["link"] for r in ok):5.1f} ms')
        for f in [f for f in ('Bs', 'Bi') if f in ok[0]['forms']]:
            print(f'  {f}: ' + ', '.join(f'{p} metal {st.mean(r["forms"][f]["pieces"][p]["metal"] for r in ok):6.1f} metallib {st.mean(r["forms"][f]["pieces"][p]["link"] for r in ok):5.1f}' for p in ('b', 'bp')))
        for f in [f for f in ('Cs', 'Ci', 'Ch', 'Co', 'Cg') if f in ok[0]['forms']]:
            print(f'  {f}: metal {st.mean(r["forms"][f]["pieces"]["u"]["metal"] for r in ok):6.1f} metallib {st.mean(r["forms"][f]["pieces"]["u"]["link"] for r in ok):5.1f} cut {st.mean(r["forms"][f]["cut"] for r in ok):5.2f}')


if __name__ == '__main__': main()
