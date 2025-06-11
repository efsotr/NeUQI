import re
import sys

from lm_eval.__main__ import cli_evaluate

import optimized_module.inject.inject_vllm_quant  # noqa: F401


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli_evaluate())
