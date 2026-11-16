"""Isolated experiment: neither candidate is imported by the production builder."""
import re


CALL = re.compile(r'fast::(min|max|clamp)\(')


def prefix_python(source):
    """Let the regex engine search a literal prefix before checking the boundary."""
    def replace(match):
        pos = match.start()
        if pos and (source[pos - 1] == '_' or source[pos - 1].isalnum()):
            return match[0]
        return 'precise::' + match[1] + '('
    return CALL.sub(replace, source)
