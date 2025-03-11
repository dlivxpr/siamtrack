import os
import torch
import sys
from siamese.core.config import cfg
from siamese.models.model_builder import ModelBuilder
from thop import profile
from thop.utils import clever_format

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))  #


def main():
    cfg.merge_from_file('')

    model = ModelBuilder()

    x = torch.randn(1, 3, 255, 255)
    zf = torch.randn(1, 3, 127, 127)

    model.template(zf)

    flop, params = profile(model, inputs=(x,), verbose=False)

    flop, params = clever_format([flop, params], "%.3f")

    print('overall flop is ', flop)

    print('overall params is ', params)


if __name__ == '__main__':
    main()