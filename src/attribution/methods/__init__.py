"""Attribution methods, all returning the same dict shape.

source_positions (n_src,), alpha (n_src,) or None, attribution (n_test, n_src)
signed toward the positive class, diagnostics, method.
"""

from .exact_loo import exact_loo_attribution
from .representer import representer_attribution
