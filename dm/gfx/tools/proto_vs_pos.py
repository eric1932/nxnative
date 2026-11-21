#!/usr/bin/env python3
# PROTOTYPE, beside proto_vs_shared.py: do today's libraries of a vertex shader and form Ch's put the same vertices at the same place? For each sampled shader it builds both
# (A: <key>_b, <key>_bp and <key>_bv as dmsh_build.py makes them; Ch: one unit cut into the same three, proto_vs_shared.py) and has proto_vs_pos (proto_vs_pos.m) run them on
# this Mac's GPU with the same random vertices, uniforms and textures:
#   gl       main0 of A's <key>_b (the reference), A's <key>_bp, Ch's <key>_b, Ch's <key>_bp (main0_p), and with integer inputs both <key>_bv (main0, main0_v)
#   z        the same with main0_z (main0_zp, main0_zv)
#   forms    main0 of A's <key>_b against Ci's and Cs's <key>_b and <key>_bp: what the other unit forms would do to the position
#   capture  main0_gs of A's <key>_b against Ch's, Ci's and Cs's: every output word of every vertex (form Ch gives this entry point an out-of-line body)
#   wide     main0_wl of A's <key>_bp against Ch's, Ci's and Cs's, the vertices as line list endpoints drawn 3 pixels wide (out of line in Ch too)
#   control  main0 with its x one unit in the last place larger, and with the depth line's 0.5 one unit larger: these must NOT agree
# In gl and z each function draws the same triangles; its depth image (what a position that differs in any bit changes) and the images of its float4 outputs (8 a pass) must be
# the reference's bit for bit. When a group does not agree, the pairs are drawn again on their own (A's <key>_b against A's <key>_bp: today's; Ch's against Ch's; Ch against A).
# A shader's first draw is a probe of 3 vertices: one whose loop the random uniforms make endless is reported and left out there.
# Usage: DM_GAME=<id> proto_vs_pos.py <library folder or image> <work folder> <proto_vs_pos binary> [spread, default 16] [largest, 4] [integer inputs, 6] [triangles, 2000]
# Writes <work>/vs_shared_pos.json; its libraries live under <work>/pos/ while a shader is tested.
import os, re, sys, json, shutil, subprocess, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..', 'tools')]
import shader_lib, dmsh_side, dmsh_pch
import proto_vs_shared as P

LOOP = re.compile(r'\b(for|while|do)\b\s*[({\n]')
NAMES = {'gl': {'b': 'main0', 'bp': 'main0', 'bv': 'main0'}, 'z': {'b': 'main0_z', 'bp': 'main0_z', 'bv': 'main0_z'}}
CH = {'gl': {'b': 'main0', 'bp': 'main0_p', 'bv': 'main0_v'}, 'z': {'b': 'main0_z', 'bp': 'main0_zp', 'bv': 'main0_zv'}}


def fragments(full, d):
    """The fragment passes that copy main0_out's float4 outputs to colour targets, 8 a pass → (their files, the locations copied, outputs of another type left out)"""
    st = re.search(r'struct main0_out\n\{\n(.*?)\n\};', full, re.S); locs, skipped = [], 0
    for ln in st[1].split('\n'):
        m = re.fullmatch(r'    (\w+) (\w+) \[\[user\(locn(\d+)\)\]\];', ln)
        if m and m[1] == 'float4': locs.append(int(m[3]))
        elif m: skipped += 1
    files = []
    for p in range(max(1, (len(locs) + 7) // 8)):
        ls = locs[p * 8:p * 8 + 8]
        src = '#include <metal_stdlib>\nusing namespace metal;\n'
        if ls: src += 'struct fin { ' + ' '.join(f'float4 v{l} [[user(locn{l})]];' for l in ls) + ' };\n'
        src += 'struct fout { ' + ' '.join(f'float4 c{i} [[color({i})]];' for i in range(8)) + ' };\n'
        src += f'fragment fout f0({"fin in [[stage_in]]" if ls else ""}) {{ fout o; ' + ' '.join(f'o.c{i} = {"in.v%d" % ls[i] if i < len(ls) else "float4(0.0)"};' for i in range(8)) + ' return o; }\n'
        files.append(os.path.join(d, f'f{p}.metal')); open(files[-1], 'w').write(src)
    return files, locs, skipped


MAIN = os.environ.get('PROTO_MAIN', 'Ch')   # the form under test (its labels stay Ch_*): PROTO_MAIN=Cg for the form the real change builds
UNITS = tuple(dict.fromkeys((MAIN, 'Co', 'Ci', 'Cs')))   # Ch is the form under test; Co (its out-of-line copy not marked noinline), Ci and Cs say what the choice of form changes


def libraries(full, d):
    """A's and the unit forms' libraries of one shader in d → ({form: {piece: path}}, whether it has integer inputs), or raises ValueError"""
    F = P.forms(full, {'A', *UNITS}); out = {'A': {}}
    for piece, src in F['A'].items():
        r = P.build(src, os.path.join(d, 'tmp'), 1, False)
        if 'error' in r: raise ValueError(f'A {piece}: {r["error"][:300]}')
        out['A'][piece] = os.path.join(d, f'A_{piece}.metallib'); open(out['A'][piece], 'wb').write(r['lib'])
    vin = 'bv' in F['A']
    for u in UNITS:
        form = u + '+v' if vin else u; out[u] = {}
        r = P.build(F[form]['u'], os.path.join(d, 'tmp'), 1, False)
        if 'error' in r: raise ValueError(f'{form}: {r["error"][:300]}')
        for piece, data in zip(('b', 'bp', 'bv'), dmsh_side.cut(r['lib'], *F['groups'][form])):
            out[u][piece] = os.path.join(d, f'{u}_{piece}.metallib'); open(out[u][piece], 'wb').write(data)
    return out, vin


def controls(full, d):
    """Two libraries that must NOT agree with A's <key>_b, so a pass means something: main0 with its position's x scaled by 1 + 2^-23 (one unit in the last place) before every
    return, and main0 with the depth line's 0.5 one unit larger → {'x': path, 'z': path}"""
    e = P.entries(full)['main0']; head, body, tail = full[:e['body']], full[e['body']:e['end']], full[e['end']:]; out = {}
    for name, new in (('x', body.replace('return out;', 'out.gl_Position.x = out.gl_Position.x * 1.00000012f;\n    return out;')),
                      ('z', body.replace('out.gl_Position.w * 0.5)', 'out.gl_Position.w * 0.50000006f)'))):
        if new == body: raise ValueError(f'control {name}: nothing to change')
        r = P.build(head + new + tail, os.path.join(d, 'tmp'), 1, False)
        if 'error' in r: raise ValueError(f'control {name}: {r["error"][:300]}')
        out[name] = os.path.join(d, f'ctl_{name}.metallib'); open(out[name], 'wb').write(r['lib'])
    return out


def run(tool, mode, count, seed, files, fns):
    """proto_vs_pos on these functions ([(label, library, entry)], the first the reference) → {label: [its lines' fields]} or {'error'}"""
    try: r = subprocess.run([tool, mode, str(count), str(seed), *files, *[f'{l}={lib}:{e}' for l, lib, e in fns]], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired: return {'error': 'timeout'}
    if r.returncode: return {'error': ('probe: ' if r.returncode == 3 else '') + (r.stderr or r.stdout)[-400:]}
    out = collections.defaultdict(list)
    for ln in r.stdout.splitlines():
        c = ln.split('\t'); rec = {}
        for x in c[1:]:
            m = re.fullmatch(r'(.*?) (-?[\d.e+x-]+|0x[0-9a-f]+|inf|nan)', x)
            if m: rec[m[1]] = int(m[2], 0) if re.fullmatch(r'-?\d+|0x[0-9a-f]+', m[2]) else float(m[2])
            else: rec.setdefault('note', x)
        out[c[0]].append(rec)
    return dict(out)


def best_seed(tool, lib, d, tries=36, n=600):
    """The seed (of tries) whose data spread the vertices widest over the screen in both depth conventions, from A's main0_gs records of n vertices: the cells of a 64 x 64
    grid over the viewport that hold a vertex inside the clip volume, the smaller of the count with main0's depth and with main0_z's → (seed, cells with main0's depth, with
    main0_z's, vertices inside with each, of n). Random uniforms send some shaders down a path that draws nothing (a particle not yet born) or to one point"""
    import numpy as np
    best = None
    for seed in range(tries):
        dump = os.path.join(d, 'cap.bin')
        try: r = subprocess.run([tool, 'capture', str(n), str(seed), f'A={lib}:main0_gs'], capture_output=True, text=True, env=dict(os.environ, PV_DUMP=dump), timeout=120)
        except subprocess.TimeoutExpired: continue
        if r.returncode or not os.path.exists(dump): continue
        w = np.fromfile(dump, dtype=np.float32).reshape(-1, 144)[:, :4].astype(np.float64); os.unlink(dump)
        x, y, z, c = w.T
        with np.errstate(invalid='ignore', divide='ignore'):
            ok = np.isfinite(w).all(axis=1) & (c > 0) & (np.abs(x) <= c) & (np.abs(y) <= c)
            cell = np.floor((x / c * 0.5 + 0.5) * 64) * 64 + np.floor((y / c * 0.5 + 0.5) * 64)
            gl, zz = ok & (z >= -c) & (z <= c), ok & (z >= 0) & (z <= c)
        cand = (min(len(np.unique(cell[gl])), len(np.unique(cell[zz]))), len(np.unique(cell[gl])) + len(np.unique(cell[zz])), seed, len(np.unique(cell[gl])), len(np.unique(cell[zz])), int(gl.sum()), int(zz.sum()))
        if best is None or cand[:2] > best[:2]: best = cand
    return (best[2], best[3], best[4], best[5], best[6], n) if best else (0, 0, 0, 0, 0, n)


def agree(rows):   # a function's passes against the reference: (depth pixels differing, output pixels differing, the largest depth difference, the largest output difference, covered pixels)
    if any('note' in r and ('failed' in r['note'] or 'error' in r['note']) for r in rows): return None
    return (max(r['depth pixels differing'] for r in rows), sum(r['output pixels differing'] for r in rows), max(r.get('largest depth difference', 0) for r in rows), rows[0]['covered'])


def main():
    lib, work, tool = shader_lib.view(sys.argv[1]), sys.argv[2], sys.argv[3]
    spread, largest, ints, tris = (int(sys.argv[i]) if len(sys.argv) > i else v for i, v in ((4, 16), (5, 4), (6, 6), (7, 2000)))
    P.PCH = dmsh_pch.header(P.METAL)
    rows = [ln.rstrip('\n').split('\t') for ln in open(os.path.join(work, 'vs_index.tsv'))]; pull = sorted((r for r in rows if r[4] != '-'), key=lambda r: r[0])
    read = lambda k: (lib / (k + '_b.metal')).read_text()
    picked = []
    def take(cands, n, why):
        got = 0
        for r in cands:
            if got >= n: break
            if any(r[0] == k for k, _ in picked): continue
            picked.append((r[0], why)); got += 1
    step = max(1, len(pull) // spread)
    take([pull[i] for i in range(0, len(pull), step)] + pull, spread, 'spread')
    seen = set(); big = [r for r in sorted(pull, key=lambda r: -int(r[1])) if not (r[1] in seen or seen.add(r[1]))]
    take(big, largest, 'largest')
    vin = [r for r in pull if r[5] == 'True']; take([vin[i] for i in range(0, len(vin), max(1, len(vin) // ints))] + vin, ints, 'integer inputs')
    results = []
    for n, (key, why) in enumerate(picked):
        d = os.path.join(work, 'pos', key); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        full = read(key); rec = {'key': key, 'why': why, 'bytes': len(full), 'loop': bool(LOOP.search(full[full.find('\nvertex main0_out main0('):full.find('\nvertex main0_out main0_z(')]))}
        try:
            libs, vin = libraries(full, d); files, locs, skipped = fragments(full, d)
            seed, c_gl, c_z, in_gl, in_z, of = best_seed(tool, libs['A']['b'], d)
            rec.update(form=MAIN + '+v' if vin else MAIN, outputs=len(locs), outputs_skipped=skipped, triangles=tris, vertices=tris * 3, seed=seed, probe={'cells gl': c_gl, 'cells z': c_z, 'inside gl': in_gl, 'inside z': in_z, 'of': of})
            for g in ('gl', 'z'):   # the test: today's libraries and form Ch's, every pipeline entry point
                fns = [(f'A_{p}', libs['A'][p], NAMES[g][p]) for p in libs['A']] + [(f'Ch_{p}', libs[MAIN][p], CH[g][p]) for p in libs[MAIN]]
                rec[g] = r = run(tool, 'raster', tris, seed, files, fns)
                if 'error' in r: continue
                if any(agree(r[l]) is None or agree(r[l])[:2] != (0, 0) for l, _, _ in fns):   # not all the reference's: each pair on its own
                    by = {l: (lb, e) for l, lb, e in fns}; pairs = [('A_b', 'A_bp'), ('Ch_b', 'Ch_bp'), ('A_b', 'Ch_b'), ('A_bp', 'Ch_bp')] + ([('A_b', 'A_bv'), ('Ch_b', 'Ch_bv'), ('A_bv', 'Ch_bv')] if vin else [])
                    rec[g + '_pairs'] = {f'{a} {b}': run(tool, 'raster', tris, seed, files, [(a, *by[a]), (b, *by[b])]) for a, b in pairs}
            # what the other forms would do to main0: Ci (the body inlined everywhere) and Cs (one out-of-line body for every entry point)
            rec['forms'] = run(tool, 'raster', tris, seed, files, [('A_b', libs['A']['b'], 'main0')] + [(f'{u}_{p}', libs[u][p], CH['gl'][p]) for u in ('Co', 'Ci', 'Cs') for p in ('b', 'bp')])
            # the entry points that form Ch gives an out-of-line body: geometry capture (every output word of every vertex) and wide lines (drawn 3 pixels wide)
            rec['capture'] = run(tool, 'capture', tris * 3, seed, [], [(f'{u}_gs', libs[u]['b'], 'main0_gs') for u in ('A', *UNITS)])
            rec['wide'] = run(tool, 'raster', tris, seed, files, [('@inputs', libs['A']['b'], 'main0')] + [(f'{u}_wl', libs[u]['bp'], 'main0_wl') for u in ('A', *UNITS)])
            if os.environ.get('PROTO_GSPOS'):   # the capture entry's positions against main0_z's: the plain source's main0_gs and the form under test's, each drawn from its capture
                nofrag = os.path.join(d, 'nofrag.metal')
                open(nofrag, 'w').write('#include <metal_stdlib>\nusing namespace metal;\nstruct fout { ' + ' '.join(f'float4 c{i} [[color({i})]];' for i in range(8)) + ' };\n'
                                        'fragment fout f0() { fout o; ' + ' '.join(f'o.c{i} = float4(0.0);' for i in range(8)) + ' return o; }\n')
                rec['gspos'] = run(tool, 'gspos', tris, seed, [nofrag], [('A_z', libs['A']['b'], 'main0_z'), ('A_gs', libs['A']['b'], 'main0_gs'), (f'{MAIN}_gs', libs[MAIN]['b'], 'main0_gs')])
            ctl = controls(full, d)
            rec['control'] = run(tool, 'raster', tris, seed, files[:1], [('A_b', libs['A']['b'], 'main0'), ('x', ctl['x'], 'main0'), ('z', ctl['z'], 'main0')])
        except (ValueError, P.Shape) as e: rec['error'] = str(e)[:400]
        shutil.rmtree(d, ignore_errors=True); results.append(rec)
        short = lambda g: {l: agree(v) for l, v in rec[g].items()} if g in rec and 'error' not in rec[g] else rec.get(g)
        cap = {l: (v[0].get('differing words'), v[0].get('position words'), v[0].get('vertices differing'), v[0].get('largest ulps')) for l, v in rec.get('capture', {}).items() if l != 'error'} or rec.get('capture')
        print(n + 1, key, why, rec.get('form'), 'seed', rec.get('seed'), rec.get('probe'), '\n   gl', short('gl'), '\n   z', short('z'), '\n   forms', short('forms'), '\n   capture', cap,
              '\n   wide', short('wide'), '\n   control', short('control'), '\n   gspos', {l: (v[0].get('depth pixels differing'), v[0].get('covered'), v[0].get('vertices differing from the first capture')) for l, v in rec.get('gspos', {}).items() if l != 'error'} or rec.get('gspos'), rec.get('error', ''), flush=True)
    shutil.rmtree(os.path.join(work, 'pos'), ignore_errors=True)
    json.dump({'library': sys.argv[1], 'rows': results}, open(os.path.join(work, 'vs_shared_pos.json'), 'w'), indent=1)
    print(f'{len(results)} shaders, {sum(r["loop"] for r in results)} with a loop in main0')


if __name__ == '__main__': main()
