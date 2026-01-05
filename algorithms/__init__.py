from .base import BlockwiseOpt, BaseBlockwiseQuantization
from .rtn import RTN as RTNQuantization
from .awq import AWQ as AWQQuantization
from .gptq import GPTQ as GPTQQuantization
from .smoothquant import SmoothQuant as SmoothQuantQuantization

# 别名
RTN = RTNQuantization
AWQ = AWQQuantization
GPTQ = GPTQQuantization
SmoothQuant = SmoothQuantQuantization

__all__ = [
    "BlockwiseOpt",
    "BaseBlockwiseQuantization",
    "RTN",
    "AWQ",
    "GPTQ",
    "SmoothQuant",
    "RTNQuantization",
    "AWQQuantization",
    "GPTQQuantization",
    "SmoothQuantQuantization",
]
