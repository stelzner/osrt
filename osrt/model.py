from torch import nn

from osrt.encoder import OSRTEncoder
from osrt.decoder import SlotMixerDecoder, SpatialBroadcastDecoder, SRTDecoder

class OSRT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = OSRTEncoder(**cfg['encoder_kwargs'])
        decoder_type = cfg['decoder']

        if decoder_type == 'spatial_broadcast':
            self.decoder = SpatialBroadcastDecoder(**cfg['decoder_kwargs'])
        elif decoder_type == 'srt':
            self.decoder = SRTDecoder(**cfg['decoder_kwargs'])
        elif decoder_type == 'slot_mixer':
            self.decoder = SlotMixerDecoder(**cfg['decoder_kwargs'])
        else:
            raise ValueError(f'Unknown decoder type: {decoder_type}')


