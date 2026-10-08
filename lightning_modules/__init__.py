try:
    from .california_datamodule import CaliforniaDataModule
    from .europe_datamodule import EuropeDataModule
except ImportError:
    pass
from .indonesia_datamodule import IndonesiaDataModule, IndonesiaDataset
