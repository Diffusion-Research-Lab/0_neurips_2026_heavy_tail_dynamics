import torch


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
REFERENCE_MODELS = {"test vs test", "test vs true sample"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
SEED = 0
N_TRIALS = 5

N_STEPS = 64
FLOW_N_STEPS = 128
FLOW_SAMPLE_STEPS = 128
BATCH_SIZE = 128
N_EPOCHS = 128
LR = 5e-4
DEPTH = 3
WIDTH = 128

ALPHA = 1.7
DIM = 15
N_TRAIN = 30_000
N_VAL = 1
N_TEST = 1_500_000
N_MMD = 10_000
SAMPLE_CHUNK_SIZE = 10_000

TCE_TAIL_PROBS = torch.logspace(-1.0, -5.0, 20, dtype=torch.float64)


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
    freq_logging=0,
)
