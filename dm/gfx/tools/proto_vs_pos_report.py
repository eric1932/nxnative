#!/usr/bin/env python3
# PROTOTYPE, beside proto_vs_pos.py: the summary of <work>/vs_shared_pos.json. Usage: proto_vs_pos_report.py <work folder>...
import os, sys, json, statistics as st


def passes(rows):   # a function's lines → (depth pixels differing, output pixels differing, covered pixels), None when it did not draw
    if not rows or any('note' in r and ('failed' in r['note'] or 'error' in r['note']) for r in rows): return None
    return max(r['depth pixels differing'] for r in rows), sum(r['output pixels differing'] for r in rows), rows[0]['covered']


def main():
    for work in sys.argv[1:]:
        d = json.load(open(os.path.join(work, 'vs_shared_pos.json'))); rows = d['rows']
        ok = [r for r in rows if 'error' not in r and all('error' not in r.get(g, {'error': 1}) for g in ('gl', 'z'))]
        print(f'\n{d["library"]}\n{len(rows)} vertex shaders ({sum(r["why"] == "spread" for r in rows)} spread, {sum(r["why"] == "largest" for r in rows)} largest sources, '
              f'{sum(r.get("form") == "Ch+v" for r in rows)} with integer inputs, {sum(r["loop"] for r in rows)} with a loop), {len(ok)} drawn; each {rows[0].get("triangles")} triangles = {rows[0].get("vertices")} vertices, a 512 x 512 target')
        for r in rows:
            if r not in ok: print('  not drawn:', r['key'], r.get('error') or {g: r.get(g, {}).get('error') for g in ('gl', 'z')})
        for g, what in (('gl', 'main0 (depth as OpenGL has it)'), ('z', 'main0_z (near = 0)')):
            cov = [passes(r[g]['A_b'])[2] for r in ok]
            print(f'  {what}: pixels covered: median {st.median(cov):.0f}, least {min(cov)}, {sum(c == 0 for c in cov)} shaders drew nothing')
            labels = [l for l in ok[0][g] if l != 'A_b'] + [l for l in ('A_bv', 'Ch_bv') if any(l in r[g] for r in ok) and l not in ok[0][g]]
            for l in dict.fromkeys(labels):
                have = [r for r in ok if l in r[g]]; res = [passes(r[g][l]) for r in have]
                bad = [(r['key'], x) for r, x in zip(have, res) if x is None or x[:2] != (0, 0)]
                print(f'    {l:6} against today\'s <key>_b: {len(have) - len(bad)} of {len(have)} shaders bit-identical in depth and in every output' + (f'; differ: {bad}' if bad else ''))
        for name, title in (('forms', 'main0 of the other unit forms against today\'s <key>_b'), ('wide', 'main0_wl (wide lines) against today\'s <key>_bp')):
            have = [r for r in ok if 'error' not in r.get(name, {'error': 1})]; ref = 'A_b' if name == 'forms' else 'A_wl'
            if not have: print(f'  {title}: not run ({[r.get(name, {}).get("error", "")[:80] for r in ok][:2]})'); continue
            cov = [passes(r[name][ref])[2] for r in have]
            print(f'  {title}: {len(have)} shaders, pixels covered: median {st.median(cov):.0f}, least {min(cov)}')
            for l in [l for l in have[0][name] if l != ref]:
                res = [passes(r[name][l]) for r in have]; bad = [(r['key'], x) for r, x in zip(have, res) if x is None or x[:2] != (0, 0)]
                print(f'    {l:6} {len(have) - len(bad)} of {len(have)} bit-identical' + (f'; differ: {bad[:6]}' if bad else ''))
        have = [r for r in ok if 'error' not in r.get('capture', {'error': 1})]
        print(f'  main0_gs (geometry capture) against today\'s: {len(have)} shaders, {sum(r["capture"]["A_gs"][0]["words"] for r in have)} output words of {sum(r["capture"]["A_gs"][0]["vertices"] for r in have)} vertices')
        for l in [l for l in have[0]['capture'] if l != 'A_gs']:
            c = [r['capture'][l][0] for r in have]; bad = [x for x in c if x['differing words']]
            print(f'    {l:6} {len(have) - len(bad)} of {len(have)} shaders bit-identical' + (f'; in the other {len(bad)}: {sum(x["differing words"] for x in bad)} words of {sum(x["vertices differing"] for x in bad)} vertices differ '
                  f'({sum(x["position words"] for x in bad)} position words), the largest by {max(x["largest ulps"] for x in bad)} units in the last place ({max(x["largest position ulps"] for x in bad)} in a position), '
                  f'{sum(x["words over 16 ulps"] for x in bad)} words by more than 16' if bad else ''))
        have = [r for r in ok if 'error' not in r.get('control', {'error': 1})]
        for l, what in (('x', 'x one unit in the last place larger'), ('z', 'the depth line\'s 0.5 one unit larger')):
            res = [passes(r['control'][l]) for r in have]; seen = [x for x in res if x and x[0]]
            print(f'  control, {what}: seen in {len(seen)} of {len(have)} shaders' + (f' (depth differs in a median of {st.median(100 * x[0] / max(1, x[2]) for x in seen):.0f}% of the covered pixels)' if seen else ''))


if __name__ == '__main__': main()
