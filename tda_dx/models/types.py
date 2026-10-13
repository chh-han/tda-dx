from beartype.typing import Union

from tda_dx.models.hubert_kmeans import HfHubertWithKmeans
from tda_dx.models.encodec_wrapper import EncodecWrapper
from tda_dx.models.stable_audio_rvq import StableAudioRVQCodec

Wav2Vec = HfHubertWithKmeans
NeuralCodec = Union[EncodecWrapper, StableAudioRVQCodec]
