#!/usr/bin/env python3
# PROTOTYPE, not part of the translation chain (dmsh_chain.FILES does not name it, no DMSH_CHAIN changes): what a vertex shader's Metal compiles would cost if its entry points
# shared one copy of its body, as a pixel shader's do (dmsh_side.py). Today <key>_b holds the body three times (main0, main0_z, main0_gs) and the pulling library <key>_bp four
# times (main0, main0_z and the wide-line bodies dm_wlb_main0, dm_wlb_main0_z), in two compiles. The forms measured here, from a built library's <key>_b.metal:
#   A    today: <key>_b as it is and dmsh_pull.variant's <key>_bp, two compiles
#   Bs   the body once in each compile, a plain static function (clang's -O1 inlines only functions marked inline, so the entry points call it), thin entry points, two compiles
#   Bi   the same with the body always_inline
#   Cs   one compile unit with every entry point of <key>_b and <key>_bp around one static body, cut into the two libraries (dmsh_side.cut)
#   Ci   the same with the body always_inline
#   Ch   the same unit, the body always_inline in the entry points pipelines are made of (main0, main0_z and their pulling and shadow-input versions) and one out-of-line
#        copy for the entry points few draws take (main0_gs, the wide-line ones), marked noinline
#   Co   Ch with that copy a plain static function (not noinline): the front end still leaves the call, and the GPU compiler may inline it
#   Cg   Co with main0_gs left as today's whole copy of the body (its capture epilogue at every return): only the wide-line entry points take the out-of-line copy
# and, for a shader with integer inputs, the shadow-input library <key>_bv (dmsh_vin.py) apart (A) or as a third piece of the unit (Cs+v, Ci+v).
# Usage: DM_GAME=<id> proto_vs_shared.py time|check|libs|index|checkall <library folder or image> <work folder> [shaders, default 40] [largest, default 8]
#   time    serial, the best of three, CPU from RUSAGE_CHILDREN: run it on an idle Mac. Writes <work>/vs_shared_time.json
#   check   J at a time (J, default 14), one compile each: which sources do not build in each form, and the bitcode sizes. Writes <work>/vs_shared_check.json
#   libs    as check, and every form's libraries stay under <work>/gpu/<form>/ for proto_vs_gpu.m (the GPU compiler's code for each form)
#   index   <work>/vs_index.tsv, every vertex shader's size and entry points, which the other modes pick their shaders from
#   checkall  every vertex shader with a pulling library in form Ch (Ch+v with integer inputs), compile and cut only, J processes. Writes <work>/vs_shared_checkall.json
# Environment: PROTO_FORMS=A,Ch (only these forms), PROTO_KEYS=<file of keys> (these shaders, not the sample), PROTO_TARGET=ios (the iPhone's compile convention, compile only),
# PROTO_TAG (a suffix for the result file's name)
# Everything it writes goes to the work folder (the game's <id>-work), nothing to the repository.
import os, re, sys, json, time, struct, hashlib, resource, subprocess, statistics, concurrent.futures
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..', 'tools')]
import shader_lib, dmsh_pull, dmsh_side, dmsh_chain, dmsh_pch, dmsh_vin

ZFIX = 'out.gl_Position.z = fma(out.gl_Position.z, 0.5, out.gl_Position.w * 0.5);'   # dmsh_entry.vs_depth
VENTRY = re.compile(r'^vertex (\S+) (\w+)\((.*)\)\n\{\n', re.M)
END = re.compile(r'\n\}(?:\n|$)')
PULLED = {'main0': 'main0_p', 'main0_z': 'main0_zp'}   # the pulling entry points' names inside one unit, where main0 and main0_z are taken
SHADOW = {'main0': 'main0_v', 'main0_z': 'main0_zv'}   # the shadow-input ones
ATTRS = {'s': '', 'i': '__attribute__((always_inline)) '}


class Shape(ValueError): pass   # a source of a form this prototype does not know


def entries(src):
    """{name: the vertex entry point's ret, params, start, body (after its opening brace's line) and end (after its closing brace's line)}"""
    out = {}
    for m in VENTRY.finditer(src):
        e = END.search(src, m.end() - 1)
        if not e: raise Shape(f'no end of {m[2]}')
        out[m[2]] = {'ret': m[1], 'params': m[3], 'start': m.start(), 'body': m.end(), 'end': e.end()}
    return out


def parts(full):
    """What the forms are made of: the entry points of <key>_b.metal, the body as main0_z has it (body_z) and as main0 has it (body_gl), main0_gs's prologue and epilogue,
    the body function's parameters and the arguments an entry point passes. Raises Shape when the source is not vs_depth's and dmsh_geometry.capture's"""
    es = entries(full)
    if any(n not in es for n in ('main0', 'main0_z', 'main0_gs')): raise Shape('not main0, main0_z and main0_gs')
    e0, ez, eg = es['main0'], es['main0_z'], es['main0_gs']
    if not e0['start'] < ez['start'] < eg['start']: raise Shape('the entry points are in another order')
    if e0['params'] != ez['params'] or e0['ret'] != 'main0_out' or ez['ret'] != 'main0_out' or eg['ret'] != 'void': raise Shape('main0 and main0_z differ in more than the depth line')
    body_gl, body_z, gs = full[e0['body']:e0['end']], full[ez['body']:ez['end']], full[eg['body']:eg['end']]
    if 'return out;' not in body_z or body_gl != body_z.replace('return out;', ZFIX + '\n    return out;'): raise Shape('main0 is not main0_z with the depth line')
    epi = re.search(r'\{ dm_cw\(.* return; \}', gs)
    if not epi: raise Shape('no capture epilogue')
    tail = body_z.replace('return out;', epi[0]).rstrip('\n')
    if not gs.rstrip('\n').endswith(tail): raise Shape('main0_gs is not the body with the capture epilogue')
    pro = gs[:len(gs.rstrip('\n')) - len(tail)]
    decl, args = [], []
    for p in dmsh_side._params(e0['params']):
        d = re.sub(r'\s*\[\[.*\]\]$', '', p); a = re.search(r'(\w+)$', d)
        if not a or 'function_constant' in p: raise Shape(f'a parameter of main0: {p[:80]}')
        decl.append(d); args.append(a[1])
    args = ', '.join(args)
    return {'es': es, 'body_gl': body_gl, 'body_z': body_z, 'pro': pro, 'epi': epi[0], 'decl': ', '.join(decl), 'args': args,
            'call_gl': f'    main0_out out = dm_vs_body({args});\n    {ZFIX}\n    return out;\n}}\n', 'call_z': f'    return dm_vs_body({args});\n}}\n',
            'other': sorted(set(es) - {'main0', 'main0_z', 'main0_gs'})}   # entry points left as they are, each with its own copy (main0_l, main0_lz, main0_discard, main0_tf)


def body_fn(P, attr): return f'static {attr}main0_out dm_vs_body({P["decl"]})\n{{\n' + P['body_z'] + '\n'


def thin(text, P, expect):   # text with each copy of the body replaced by a call of dm_vs_body; expect: how many copies of each there must be
    if text.count(P['body_gl']) != expect or text.count(P['body_z']) != expect: raise Shape(f'{text.count(P["body_gl"])} and {text.count(P["body_z"])} copies of the body, not {expect}')
    return text.replace(P['body_gl'], P['call_gl']).replace(P['body_z'], P['call_z'])


OUTER = 'dm_vs_body_o'   # form Ch: the body out of line, for the entry points few draws take (main0_gs, the wide-line ones), while main0 and main0_z inline it


NOINLINE = '__attribute__((noinline)) '   # form Ch marks the out-of-line copy noinline; form Co leaves it a plain static function, which clang's -O1 does not inline either


def outer(P, attr=NOINLINE): return f'static {attr}main0_out {OUTER}({P["decl"]})\n{{\n    return dm_vs_body({P["args"]});\n}}\n\n'


def unit_b(full, P, attr, rare='dm_vs_body', outer_attr=NOINLINE, gs_copy=False):
    """<key>_b's source with the body once: dm_vs_body, then main0 (the body, then the depth line), main0_z (the body) and main0_gs (its prologue, the body, its epilogue).
    rare: the function main0_gs calls (OUTER: an out-of-line copy of the body, defined here)"""
    e0, ez, eg = (P['es'][n] for n in ('main0', 'main0_z', 'main0_gs'))
    gs = full[eg['body']:eg['end']] if gs_copy else P['pro'] + f'    main0_out out = {rare}({P["args"]});\n    {P["epi"]}\n}}\n'
    return (full[:e0['start']] + body_fn(P, attr) + (outer(P, outer_attr) if rare == OUTER else '') + full[e0['start']:e0['body']] + P['call_gl'] + full[e0['end']:ez['body']] + P['call_z']
            + full[ez['end']:eg['body']] + gs + full[eg['end']:])


def unit_bp(a_bp, info, P, attr):
    """dmsh_pull.variant's source with the body once: its main0 and main0_z keep their prologue (the inputs pulled), and the wide-line bodies dm_wlb_<entry> call dm_vs_body too"""
    s = thin(a_bp, P, 2 if 'entry_wl' in info else 1)
    i = re.search(r'^vertex ', s, re.M).start()
    return s[:i] + body_fn(P, attr) + s[i:]


def one_copy(text, P):   # a single function's text with its copy of the body (either form) replaced by the call
    n = text.count(P['body_gl']) + text.count(P['body_z'])
    if n != 1: raise Shape(f'{n} copies of the body in one function')
    return text.replace(P['body_gl'], P['call_gl']).replace(P['body_z'], P['call_z'])


def unit_c(full, a_bp, info, P, attr, a_bv=None, rare='dm_vs_body', outer_attr=NOINLINE, gs_copy=False):
    """One compile unit: unit_b (main0_in keeps its attributes, and the pulling entry points fill a local of that type), dmsh_pull's fetch code, the pulling entry points as
    main0_p / main0_zp and the wide-line ones, and with a_bv the shadow-input entry points as main0_v / main0_zv. → (source, the pulling library's names, the shadow one's)"""
    b = unit_b(full, P, attr, rare, outer_attr, gs_copy); wl = 'entry_wl' in info
    ui = b.index('using namespace metal;') + len('using namespace metal;')
    b = b[:ui] + '\n\n' + dmsh_pull.FETCH + (dmsh_pull.WL_MSL if wl else '') + b[ui:]
    pe = entries(a_bp); add = ''; names = set()
    for n, new in PULLED.items():
        if n not in pe: continue
        e = pe[n]; add += '\n' + one_copy(a_bp[e['start']:e['end']].replace(f'vertex main0_out {n}(', f'vertex main0_out {new}(', 1), P); names.add(new)
    if wl:
        add += thin(a_bp[a_bp.index('\nstatic inline main0_in dm_wl_in('):], P, 1).replace('dm_vs_body(', rare + '('); names |= {'main0_wl', 'main0_z_wl'}
    vnames = set()
    if a_bv:   # main0_in_v (the stage input with the shadow members) and the two entry points whose prologue ORs them into main0_in
        st = re.search(r'\nstruct main0_in_v\n\{\n.*?\n\};', a_bv, re.S); ve = entries(a_bv)
        if not st: raise Shape('no main0_in_v')
        add += '\n' + st[0] + '\n'
        for n, new in SHADOW.items():
            e = ve[n]; add += '\n' + one_copy(a_bv[e['start']:e['end']].replace(f'vertex main0_out {n}(', f'vertex main0_out {new}(', 1), P); vnames.add(new)
    return b + add, names, vnames


def unit_bv(a_bv, P, attr):   # dmsh_vin.variant's source with the body once (its main0_gs too, as unit_b's; its other entry points stay as they are)
    es = entries(a_bv); s = a_bv
    e0, ez, eg = (es[n] for n in ('main0', 'main0_z', 'main0_gs'))
    return (s[:e0['start']] + body_fn(P, attr) + one_copy(s[e0['start']:e0['end']], P) + s[e0['end']:ez['start']] + one_copy(s[ez['start']:ez['end']], P) + s[ez['end']:eg['body']]
            + P['pro'] + f'    main0_out out = dm_vs_body({P["args"]});\n    {P["epi"]}\n}}\n' + s[eg['end']:])


WANT = set(filter(None, os.environ.get('PROTO_FORMS', '').split(','))) or None   # PROTO_FORMS=A,Ch: only these forms (one with integer inputs also as <form>+v)


def forms(full, want=None):
    """{form: {piece: source}} and the unit forms' cut groups, for the forms in want (default: PROTO_FORMS, else all of them). Raises Shape"""
    want = want or WANT; on = lambda f: not want or f in want
    P = parts(full)
    a_bp, info = dmsh_pull.variant(full)
    if not a_bp: raise Shape(f'no pulling variant ({info})')
    a_bv, shadows = dmsh_vin.variant(full)
    out = {'groups': {}, 'other': P['other'], 'wl': 'entry_wl' in info, 'vin': bool(a_bv)}
    if on('A'): out['A'] = {'b': full, 'bp': a_bp, **({'bv': a_bv} if a_bv else {})}
    def unit(form, attr, rare='dm_vs_body', outer_attr=NOINLINE, gs_copy=False):
        src, names, _ = unit_c(full, a_bp, info, P, attr, rare=rare, outer_attr=outer_attr, gs_copy=gs_copy)
        out[form] = {'u': src}; out['groups'][form] = [sorted(names)]
        if a_bv:
            src, names, vnames = unit_c(full, a_bp, info, P, attr, a_bv, rare=rare, outer_attr=outer_attr, gs_copy=gs_copy)
            out[form + '+v'] = {'u': src}; out['groups'][form + '+v'] = [sorted(names), sorted(vnames)]
    for k, attr in ATTRS.items():
        if on('B' + k): out['B' + k] = {'b': unit_b(full, P, attr), 'bp': unit_bp(a_bp, info, P, attr), **({'bv': unit_bv(a_bv, P, attr)} if a_bv else {})}
        if on('C' + k): unit('C' + k, attr)
    if on('Ch'): unit('Ch', ATTRS['i'], OUTER)
    if on('Co'): unit('Co', ATTRS['i'], OUTER, '')
    if on('Cg'): unit('Cg', ATTRS['i'], OUTER, '', True)
    return out


def xo(*a): return subprocess.run(['xcrun', '-sdk', 'macosx', *a], capture_output=True, text=True, check=True).stdout.strip()
TARGET = os.environ.get('PROTO_TARGET', 'mac')   # ios: the iPhone's compile convention, compile only (ios_metal.py's METAL, DEPTH and METALLIB, through xcrun as ios_metal.compile runs them)
if TARGET == 'ios':
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'ios', 'tools')); import ios_metal
    METAL = [*ios_metal.METAL, *ios_metal.DEPTH]; LINK = list(ios_metal.METALLIB)
else:
    METAL = [xo('-f', 'metal'), '-isysroot', xo('--show-sdk-path'), *dmsh_chain.MAC_METAL[4:], '-fbracket-depth=1024']   # dmsh_build.py MAC_RUN + METAL_DEPTH
    LINK = [xo('-f', 'metallib')]
PCH = None


def cpu(): r = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime


def fns(lib):
    """{function: (its bitcode's size, a digest of it)} of a Metal library"""
    s = struct.unpack_from('<QQQQQQQQ', lib, 0x18); fo, bo = s[0], s[6]
    n = struct.unpack_from('<I', lib, fo)[0]; p = fo + 4; out = {}
    for _ in range(n):
        size = struct.unpack_from('<I', lib, p)[0]; t = dict(dmsh_side._tags(lib, p + 4, p + size)); p += size
        off = struct.unpack('<QQQ', t[b'OFFT'])[2]; sz = struct.unpack('<Q', t[b'MDSZ'])[0]
        out[t[b'NAME'].rstrip(b'\0').decode()] = (sz, hashlib.sha256(lib[bo + off:bo + off + sz]).hexdigest()[:16])
    return out


def build(src, stem, runs, timed):
    """src → a library: {'metal': CPU ms of the compile, 'link': of metallib (the best of runs, when timed), 'air': the .air's size, 'lib': the library's bytes} or {'error'}"""
    air, lib = stem + '.air', stem + '.metallib'; tm, tl = [], []
    try:
        for _ in range(runs):
            c0 = cpu() if timed else 0
            r = subprocess.run([*METAL, *(['-include-pch', PCH] if PCH else []), '-x', 'metal', '-', '-o', air], input=src.encode(), capture_output=True)
            if r.returncode: return {'error': r.stderr.decode(errors='replace')[:600]}
            c1 = cpu() if timed else 0
            r = subprocess.run([*LINK, air, '-o', lib], capture_output=True)
            if r.returncode: return {'error': 'metallib: ' + r.stderr.decode(errors='replace')[:600]}
            tm.append(1000 * (c1 - c0)); tl.append(1000 * ((cpu() if timed else 0) - c1))
        with open(lib, 'rb') as f: data = f.read()
        return {'metal': min(tm), 'link': min(tl), 'air': os.path.getsize(air), 'lib': data}
    finally:
        for p in (air, lib):
            if os.path.exists(p): os.unlink(p)


KEEP = False   # mode libs: every form's libraries stay, as <work>/gpu/<form>/<key>_<b|bp|bv>.metallib, for proto_vs_gpu


def keep(work, form, key, piece, data):
    if not KEEP: return
    d = os.path.join(work, 'gpu', form); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'{key}_{piece}.metallib'), 'wb') as f: f.write(data)


def measure(key, full, work, runs, timed):
    """One shader in every form → a row: per form its CPU ms (metal, link, cut), the .air bytes, each library's bytes and its functions' bitcode sizes and digests; or why not"""
    try: F = forms(full)
    except Shape as e: return {'key': key, 'shape': str(e)}
    row = {'key': key, 'bytes': len(full), 'other': F['other'], 'wl': F['wl'], 'vin': F['vin'], 'forms': {}}
    for form in [f for f in F if f[0] in 'ABC']:
        rec = {'metal': 0.0, 'link': 0.0, 'cut': 0.0, 'air': 0, 'compiles': 0, 'libs': {}, 'pieces': {}}
        for piece, src in F[form].items():
            r = build(src, os.path.join(work, f'{key}-{form}-{piece}-{os.getpid()}'), runs, timed)
            if 'error' in r:
                rec['error'] = f'{piece}: {r["error"]}'
                with open(os.path.join(work, f'fail-{key}-{form}-{piece}.metal'), 'w') as f: f.write(src)
                break
            rec['metal'] += r['metal']; rec['link'] += r['link']; rec['air'] += r['air']; rec['compiles'] += 1; rec['pieces'][piece] = {'metal': r['metal'], 'link': r['link'], 'air': r['air']}
            if piece != 'u':
                rec['libs'][piece] = {'bytes': len(r['lib']), 'fns': fns(r['lib'])}; keep(work, form, key, piece, r['lib']); continue
            best = None
            for _ in range(runs):
                t0 = time.process_time(); pieces = dmsh_side.cut(r['lib'], *F['groups'][form]); t = 1000 * (time.process_time() - t0); best = t if best is None else min(best, t)
            rec['cut'] = best
            for name, data in zip(('b', 'bp', 'bv'), pieces): rec['libs'][name] = {'bytes': len(data), 'fns': fns(data)}; keep(work, form, key, name, data)
        row['forms'][form] = rec
    return row


def pick(index, n, largest):
    """n keys spread evenly over the vertex shaders with a pulling library, then the largest sources not among them (one per distinct size)"""
    rows = [ln.rstrip('\n').split('\t') for ln in open(index)]
    keys = sorted(r[0] for r in rows if r[4] != '-')
    out = keys[::max(1, len(keys) // n)][:n]; seen = set()
    for r in sorted((r for r in rows if r[4] != '-'), key=lambda r: -int(r[1])):
        if len(out) >= n + largest: break
        if r[0] in out or r[1] in seen: continue
        out.append(r[0]); seen.add(r[1])
    return out, len(keys)


ENTRY = re.compile(r'^vertex\s+\S+\s+(\w+)\(', re.M)


def index(lib, work):
    """<work>/vs_index.tsv: every vertex shader of a library: key, source bytes, lines, entry points, the reflection's pulling entry names (- for none), integer inputs, chain"""
    rows = []
    for p in sorted(lib.glob('*_b.json'), key=lambda p: p.name):
        r = json.loads(p.read_text())
        if r.get('stage') != 'vs': continue
        k = p.name[:-7]; src = (lib / (k + '_b.metal')).read_text(); pull = r.get('pull')
        rows.append((k, len(src), src.count('\n'), ' '.join(ENTRY.findall(src)), ' '.join(x for x in sorted(pull) if x.startswith('entry')) if pull else '-', bool(r.get('vin')), r.get('chain')))
    with open(os.path.join(work, 'vs_index.tsv'), 'w') as f: f.write(''.join('\t'.join(map(str, r)) + '\n' for r in rows))
    print(f'{len(rows)} vertex shaders, {sum(r[4] != "-" for r in rows)} with a pulling library, chain {rows[0][-1] if rows else "-"}')


_LIB = None
UNIT = os.environ.get('PROTO_UNIT', 'Co')   # the unit form checkall builds


def _start(lib, pch):   # a checkall worker: its own view of the library, the parent's precompiled header
    global _LIB, PCH
    _LIB, PCH = shader_lib.view(lib), pch


def check_one(arg):
    """One shader in form Ch (Ch+v with integer inputs), compile only: (key, 'shape' | 'same' | 'ok' | 'fail', detail, entry points left as copies, wide lines, integer inputs).
    A unit another shader already compiled (the same text) is not compiled again, as the Metal cache would not"""
    key, work = arg
    full = (_LIB / (key + '_b.metal')).read_text()
    try: F = forms(full, {UNIT})
    except Shape as e: return key, 'shape', str(e), [], False, False
    form = UNIT + '+v' if UNIT + '+v' in F else UNIT; src = F[form]['u']; d = hashlib.sha256(src.encode()).hexdigest()[:32]
    try: os.close(os.open(os.path.join(work, 'units', d), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError: return key, 'same', d, F['other'], F['wl'], F['vin']
    r = build(src, os.path.join(work, 'units', f'{d}-{os.getpid()}'), 1, False)
    if 'error' not in r:
        try:
            pieces = dmsh_side.cut(r['lib'], *F['groups'][form]); want = [{'main0', 'main0_z', 'main0_gs'} | set(F['other']), *map(set, F['groups'][form])]
            if [set(fns(x)) for x in pieces] != want: r = {'error': f'the cut gave {[sorted(fns(x)) for x in pieces]}'}
        except ValueError as e: r = {'error': f'cut: {e}'}
    if 'error' in r:
        with open(os.path.join(work, f'fail-{key}-{form}.metal'), 'w') as f: f.write(src)
        return key, 'fail', r['error'], F['other'], F['wl'], F['vin']
    return key, 'ok', d, F['other'], F['wl'], F['vin']


def checkall(libpath, work):
    """Every vertex shader with a pulling library, in form Ch, compile only. Writes <work>/vs_shared_checkall.json"""
    import collections, shutil
    rows = [ln.rstrip('\n').split('\t') for ln in open(os.path.join(work, 'vs_index.tsv'))]; keys = sorted(r[0] for r in rows if r[4] != '-')
    units = os.path.join(work, 'units'); shutil.rmtree(units, ignore_errors=True); os.makedirs(units)
    t0 = time.time(); c = collections.Counter(); other = collections.Counter(); bad = []
    with concurrent.futures.ProcessPoolExecutor(int(os.environ.get('J', '16')), initializer=_start, initargs=(libpath, PCH)) as ex:
        for key, what, detail, oth, wl, vin in ex.map(check_one, [(k, work) for k in keys], chunksize=8):
            c[what] += 1; c['wide lines'] += wl; c['integer inputs'] += vin
            if what != 'shape': other[' '.join(oth) or '(none)'] += 1
            if what in ('shape', 'fail'): bad.append((key, what, detail[:500]))
    shutil.rmtree(units, ignore_errors=True)
    sets = collections.Counter((r[3], r[4] != '-') for r in rows)
    out = {'library': libpath, 'target': TARGET, 'form': UNIT, 'vertex shaders': len(rows), 'with a pulling library': len(keys), 'counts': dict(c), 'entry points left as copies': dict(other), 'bad': bad,
           'entry point sets': [[e, pull, n] for (e, pull), n in sets.most_common()], 'seconds': round(time.time() - t0)}
    json.dump(out, open(os.path.join(work, 'vs_shared_checkall.json'), 'w'), indent=1)
    print(f'checkall ({TARGET}, form {UNIT}): {len(keys)} vertex shaders with a pulling library of {len(rows)}, {time.time() - t0:.0f} s: {c["ok"]} units built, {c["same"]} shaders shared a unit, '
          f'{c["fail"]} do not build, {c["shape"]} of a form the prototype does not know; entry points left as copies: {dict(other)}')
    for b in bad[:12]: print('  ', b[0], b[1], b[2][:300].replace('\n', ' | '))


def main():
    global PCH, KEEP
    mode, lib, work = sys.argv[1], shader_lib.view(sys.argv[2]), sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 40; largest = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.makedirs(work, exist_ok=True)
    if mode == 'index': return index(lib, work)
    PCH = dmsh_pch.header(METAL)
    if mode == 'checkall': return checkall(sys.argv[2], work)
    keys, total = pick(os.path.join(work, 'vs_index.tsv'), n, largest)
    if os.environ.get('PROTO_KEYS'): keys = open(os.environ['PROTO_KEYS']).read().split()   # these shaders, not the sample
    read = lambda k: (lib / (k + '_b.metal')).read_text()
    t0 = time.time()
    KEEP = mode == 'libs'
    if mode == 'time': rows = [measure(k, read(k), work, 3, True) for k in keys]
    else:
        with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('J', '14'))) as ex: rows = list(ex.map(lambda k: measure(k, read(k), work, 1, False), keys))
    name = f'vs_shared_{mode}' + ('' if TARGET == 'mac' else '_' + TARGET) + (os.environ.get('PROTO_TAG') and '_' + os.environ['PROTO_TAG'] or '')
    json.dump({'library': str(sys.argv[2]), 'with_pull': total, 'target': TARGET, 'metal': METAL[3:], 'rows': rows}, open(os.path.join(work, name + '.json'), 'w'))
    print(f'{mode} ({TARGET}): {len(rows)} vertex shaders of {total} with a pulling library, {time.time() - t0:.0f} s; {sum("shape" in r for r in rows)} of an unknown form')


if __name__ == '__main__': main()
