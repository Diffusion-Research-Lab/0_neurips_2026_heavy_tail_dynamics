import math
import torch


MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
REFERENCE_MODELS = {"test vs true sample"}

DEVICE = torch.device("cpu")
DTYPE = torch.float32
SEED = 0
N_TRIALS = 2

N_STEPS = 64
N_EPOCHS = 512
BATCH_SIZE = 128
LR = 5e-4
DEPTH = 3
WIDTH = 128

ALPHA = 1.7
DIM = 2
N_TRAIN = 5_000
N_VAL = 1
N_TEST = 100_000
N_MMD = 2_000

TCE_TAIL_PROBS = torch.logspace(math.log10(0.5), -4.0, 30, dtype=torch.float64)
TCE_QUANTILES = 1.0 - TCE_TAIL_PROBS


def _tce_label(prob):
    quantile = 100.0 * (1.0 - float(prob))
    precision = 2 if quantile < 99.0 else 4
    label = f"{quantile:.{precision}f}".rstrip("0").rstrip(".")
    return f"TCE({label})"


TCE_COLUMNS = [_tce_label(prob) for prob in TCE_TAIL_PROBS]

TRAIN_KWARGS = dict(
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    lr=LR,
    device=DEVICE,
    use_adamw=False,
    lr_schedule="constant",
)
