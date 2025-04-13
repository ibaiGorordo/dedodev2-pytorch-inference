import torch
import torch.nn as nn

from dedodev2.dedode_detector import DeDoDeDetector
from dedodev2.decoder import ConvRefiner, Decoder
from dedodev2.encoder import VGG19, VGG
from dedodev2.dedode_descriptor import DeDoDeDescriptor
from dedodev2.utils import get_best_device


def dedode_detector_L(device = get_best_device(), weights = None, remove_borders = False):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url("https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth", map_location = device)
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp_dtype = torch.float16#torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
            "4": ConvRefiner(
                256+256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "2": ConvRefiner(
                128+128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp = amp, amp_dtype = amp_dtype)
    decoder = Decoder(conv_refiner)
    model = DeDoDeDetector(encoder = encoder, decoder = decoder, remove_borders = remove_borders).to(device)
    model.eval()
    if weights is not None:
        model.load_state_dict(weights)
    return model

def dedode_descriptor_B(device = get_best_device(), weights = None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url("https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth", map_location=device)
    NUM_PROTOTYPES = 256 # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16#torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
            "4": ConvRefiner(
                256+256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "2": ConvRefiner(
                128+128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
        }
    )
    encoder = VGG(size = "19", amp = amp, amp_dtype = amp_dtype)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder = encoder, decoder = decoder).to(device)
    model.eval()
    if weights is not None:
        model.load_state_dict(weights)
    return model