#!/usr/bin/env python3
# PROTOTYPE, beside proto_vs_shared.py: does the offline pipeline path (pso_offline.py: its plan, its script, real_libraries with needs_cut, metal-tt) take a vertex shader's
# libraries when they are cut from one compile unit? For a handful of the profile's shader pairs it makes two small library folders named as the game's (b-<chain>): one with
# today's <key>_b and <key>_bp (form A), one with the unit form's cut pieces, and the pixel shaders' libraries and every reflection copied from the built library. Then it runs
# pso_offline.py on each, for the iPhone (g18p) and for this Mac's GPU, in both depth modes (gl takes the pulling library's first function, zero its second).
# No host change is needed for this: pso_offline.py reads the vertex entry point's name from the reflection's `entry` / `entry_z`, so the unit form's reflection names the pulling
# entry points there (main0_p, main0_zp); the real change would have it read `pull.entry`.
# Usage: DM_GAME=<id> proto_vs_tt.py <library folder or image> <work folder> <shader_pairs.tsv> [pairs, default 8] [unit form, default Co]
import os, re, sys, json, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..', 'tools')]
import shader_lib, dmsh_side, dmsh_pch
import proto_vs_shared as P

ARCH_M = '#import <Metal/Metal.h>\nint main(void) { @autoreleasepool { printf("%s\\n", MTLCreateSystemDefaultDevice().architecture.name.UTF8String); } return 0; }\n'


def mac_arch(work):
    src, exe = os.path.join(work, 'mtl_arch.m'), os.path.join(work, 'mtl_arch'); open(src, 'w').write(ARCH_M)
    subprocess.run(['xcrun', 'clang', '-fobjc-arc', '-framework', 'Metal', '-framework', 'Foundation', src, '-o', exe], check=True)
    return subprocess.run([exe], capture_output=True, text=True, check=True).stdout.strip()


def main():
    lib, work, pairs_file = shader_lib.view(sys.argv[1]), sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 8; unit = sys.argv[5] if len(sys.argv) > 5 else 'Co'
    chain = shader_lib.name_of(lib)[2:]; tt = os.path.join(work, 'tt'); shutil.rmtree(tt, ignore_errors=True); os.makedirs(tt)
    P.PCH = dmsh_pch.header(P.METAL)
    pairs = [tuple(ln.split()[:2]) for ln in open(pairs_file) if ln.strip() and not ln.startswith('#')]
    names = {p.name for p in lib.glob('*_b.metallib')}
    picked, sampled = [], 0
    for want_border in (True, False):   # half whose vertex shader samples (it declares the border corrections' constant, so metal-tt specializes it), half that do not
        for vk, fk in pairs[::max(1, len(pairs) // 400)]:
            if len(picked) >= (n // 2 if want_border else n): break
            if any(vk == v for v, _, _ in picked) or vk + '_b.metallib' not in names or fk + '_b.metallib' not in names: continue
            vj = json.loads((lib / (vk + '_b.json')).read_text())
            if not isinstance(vj.get('pull'), dict) or vj.get('layer_out'): continue
            full = (lib / (vk + '_b.metal')).read_text()
            if ('dm_border_off' in full) != want_border: continue
            picked.append((vk, fk, full)); sampled += want_border
    dirs = {f: os.path.join(tt, f, 'b-' + chain) for f in ('A', unit)}
    for d in dirs.values(): os.makedirs(d)
    for vk, fk, full in picked:
        F = P.forms(full, {'A', unit}); vj = json.loads((lib / (vk + '_b.json')).read_text()); form = unit + '+v' if unit + '+v' in F else unit
        for piece in ('b', 'bp'):
            r = P.build(F['A'][piece], os.path.join(tt, 'tmp'), 1, False); open(os.path.join(dirs['A'], f'{vk}_{piece}.metallib'), 'wb').write(r['lib'])
        r = P.build(F[form]['u'], os.path.join(tt, 'tmp'), 1, False)
        for piece, data in zip(('b', 'bp', 'bv'), dmsh_side.cut(r['lib'], *F['groups'][form])): open(os.path.join(dirs[unit], f'{vk}_{piece}.metallib'), 'wb').write(data)
        json.dump(vj, open(os.path.join(dirs['A'], vk + '_b.json'), 'w'))
        json.dump(dict(vj, entry='main0_p', entry_z='main0_zp'), open(os.path.join(dirs[unit], vk + '_b.json'), 'w'))   # the pulling entry points' names in the unit
        for d in dirs.values():
            for name in (fk + '_b.json', fk + '_b.metallib'):
                with open(os.path.join(d, name), 'wb') as f: f.write((lib / name).read_bytes())
    pf = os.path.join(tt, 'pairs.tsv'); open(pf, 'w').write(''.join(f'{vk}\t{fk}\n' for vk, fk, _ in picked))
    import pso_offline as po   # how many of the unit's pulling libraries pso_offline.py has to cut again: a specialized function that is not its library's first (needs_cut)
    again = {e: sum(po.needs_cut(po.constants(os.path.join(dirs[unit], vk + '_bp.metallib')), e) for vk, _, _ in picked) for e in ('main0_p', 'main0_zp')}
    print(f'{len(picked)} pairs ({sampled} whose vertex shader declares the border constant), chain {chain}, unit form {unit}; pulling entry points specialized from a library of their own (needs_cut): {again}')
    for platform, gpu in (('ios', 'applegpu_g18p'), ('macos', mac_arch(tt))):
        for depth in ('gl', 'zero'):
            for f, d in dirs.items():
                out, w = os.path.join(tt, f, f'out-{platform}-{depth}'), os.path.join(tt, f, f'work-{platform}-{depth}')
                r = subprocess.run([sys.executable, os.path.join(HERE, 'pso_offline.py'), '--msl', d, '--out', out, '--work', w, '--pairs', pf, '--gpu', gpu, '--platform', platform, '--depth', depth, '--no-compute'],
                                   capture_output=True, text=True)
                arch = os.path.join(out, gpu + '.mtl4a'); size = os.path.getsize(arch) if os.path.exists(arch) else 0
                plan = next((ln for ln in r.stdout.splitlines() if ' bases, ' in ln), '')
                cuts = len([x for x in os.listdir(os.path.join(w, 'libraries')) if '@' in x]) if os.path.isdir(os.path.join(w, 'libraries')) else '?'
                red = [ln for ln in (r.stdout + r.stderr).splitlines() if '[RED]' in ln or 'error' in ln.lower()]
                print(f'  {platform} {gpu} depth {depth} form {f}: exit {r.returncode}, archive {size} bytes; {plan[plan.find(":") + 2:][:150]}' + (f'\n    {red[:3]}' if red else ''))
                if r.returncode: print('    ' + (r.stdout + r.stderr)[-600:].replace('\n', '\n    '))
    shutil.rmtree(tt, ignore_errors=True)


if __name__ == '__main__': main()
