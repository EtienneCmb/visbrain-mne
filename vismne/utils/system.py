""""""
import sys

# Py3k compat
if sys.version[0] == '2':
    string_types = basestring  # noqa, analysis:ignore
else:
    string_types = str
