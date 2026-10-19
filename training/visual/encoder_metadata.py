"""Torch-free identity of the existing frozen visual cache format."""


def encoder_metadata(weights_sha256):
    return dict(architecture='resnet18', weights='IMAGENET1K_V1', weights_sha256=weights_sha256,
                output_dim=512, preprocessing=dict(version='driver-roi-letterbox-v1', size=[224,224],
                    color='RGB', interpolation='bilinear', padding=[0,0,0],
                    mean=[.485,.456,.406], std=[.229,.224,.225]),
                sampling=dict(hz=25, offsets=[12,24], max_age_ms=80))
