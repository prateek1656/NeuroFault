"""Must run before torch or sklearn (pulled in transitively by aihwkit) load
their C extensions: both bundle their own libomp.dylib, and macOS aborts the
process on the second dlopen unless this is set. This doesn't
silently change numerics for this project's (single-process, CPU-only) use.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
