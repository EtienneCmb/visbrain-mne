""""""
import sys

# See : https://github.com/nipy/PySurfer/blob/master/surfer/utils.py#L25
# Py3k compat
if sys.version[0] == '2':
    string_types = basestring  # noqa
else:
    string_types = str
