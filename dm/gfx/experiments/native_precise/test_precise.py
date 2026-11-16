"""Differential tests against the production rewrite, including Unicode boundaries."""
import ast
from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'dm/gfx/tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmsh_rewrite import precise as upstream  # noqa: E402
from candidates import prefix_python  # noqa: E402
from _precise_native import precise as native  # noqa: E402


class Precise(unittest.TestCase):
    def equal(self, source):
        expected = upstream(source)
        self.assertEqual(prefix_python(source), expected)
        self.assertEqual(native(source), expected)

    def test_boundaries_and_suffixes(self):
        prefixes = ('', ' ', '\n', '.', ':', '_', 'a', '0', '\x00', '\u00e9', '\u4e2d',
                    '\u00b2', '\u0301', '\U0001f642', '\U00010400', '\ud800')
        calls = ('min(', 'max(', 'clamp(', 'minimum(', 'min (', 'Min(', '', 'min', 'clamp')
        for before, call, after in itertools.product(prefixes, calls, ('', 'x)', '\x00', '\U0001f642')):
            with self.subTest(before=before, call=call, after=after): self.equal(before + 'fast::' + call + after)

    def test_adjacent_and_nested(self):
        for source in ('', 'fast::min(fast::max(a,b),fast::clamp(c,d,e))',
                       'fast::fast::min(', 'fast::min(fast::max(', 'fast::clamp(' * 10000,
                       '// fast::min(a,b)\n"fast::max(x,y)"', 'x' * 500000 + ' fast::max(a,b)'):
            self.equal(source)

    def test_randomized_differential(self):
        rng = random.Random(1932)
        tokens = ['fast::', 'min(', 'max(', 'clamp(', 'fast::min(', 'fast::max(', 'fast::clamp(',
                  'xfast::min(', '_', 'a', '9', ' ', ')', '\n', '\x00', '\u00e9', '\u0301', '\U0001f642']
        for _ in range(10000):
            self.equal(''.join(rng.choices(tokens, k=rng.randrange(0, 80))))

    def test_existing_synthetic_fixture_strings(self):
        # AST reading avoids running the GPU/compiler tests, and never reads game files.
        checked = 0
        for path in sorted((ROOT / 'dm/gfx/tests').glob('test_dmsh_*.py')):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.equal(node.value)
                    checked += 1
        self.assertGreater(checked, 1000)

    def test_repeated_and_threaded_calls(self):
        source = ('fast::min(x, y); \u00e9fast::max(a,b); ' * 1000)
        expected = upstream(source)
        with ThreadPoolExecutor(4) as pool:
            self.assertTrue(all(value == expected for value in pool.map(native, [source] * 100)))

    def test_wrong_input_rejected(self):
        for source in (None, 123, b'fast::min(x,y)', [], {}):
            with self.assertRaises(TypeError): native(source)


if __name__ == '__main__': unittest.main()
