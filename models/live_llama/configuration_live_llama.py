from .mod_llama import MoDLlamaConfig
from ..configuration_live import LiveConfigMixin

class LiveLlamaConfig(MoDLlamaConfig, LiveConfigMixin): 
    pass