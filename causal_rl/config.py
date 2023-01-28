from typing import Literal
import torch
from pathlib import Path


TOP_LEVEL = Path(__file__).parent.parent
DATA = TOP_LEVEL / "data"
DATA.mkdir(exist_ok=True, parents=True)
LOGS = TOP_LEVEL / "logs"
PLOTS = TOP_LEVEL / "plots"
MODELS = TOP_LEVEL / "models"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


TrackerKey = Literal["wandb", "console"]
