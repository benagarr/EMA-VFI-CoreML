import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
import coremltools as ct
#import coremltools.optimize.coreml as cto

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
#model.device()


print(f'=========================Start Generating=========================')

#I0 = cv2.imread('example/img1.jpg')
#I2 = cv2.imread('example/img2.jpg')

#I0 = cv2.imread('example/jp1-700.jpg')
#I2 = cv2.imread('example/jp2-700.jpg')

I0 = cv2.imread('example/KC1-512.jpg')
I2 = cv2.imread('example/KC2-512.jpg')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

imgs = torch.cat((I0_, I2_), 1)
check_trace = False
traced_model = torch.jit.trace(model.net, (I0_, I2_), check_trace=check_trace)
#model_from_torch = ct.convert(traced_model,
#                              convert_to="mlprogram",
#                              inputs=[ct.TensorType(name="input",
#                                                    shape=imgs.shape)])

#scale = 1/(0.226*255.0)
scale = 1 / 255.0
#bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

model_from_torch = ct.convert(traced_model,
                              convert_to="mlprogram",
                              compute_precision=ct.precision.FLOAT16,
                              inputs=[ct.ImageType(name="input1",
                                                    shape=I0_.shape,
                                                    color_layout=ct.colorlayout.RGB,
                                                    scale=scale),
                                                    ct.ImageType(name="input2",
                                                    shape=I0_.shape,
                                                    color_layout=ct.colorlayout.RGB,
                                                    scale=scale)],
                              outputs=[ct.ImageType(name="pred", scale=scale)])

# save without compressing
model_from_torch.save('/content/result.mlpackage')


# Compress it
#model_compressed = ct.compression_utils.affine_quantize_weights(model_from_torch)
#model_compressed.save('/content/result.mlpackage')

# Compress newer version (beta)
# define op config
#op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=6)

# define optimization config by applying the op config globally to all ops
#config = cto.OptimizationConfig(global_config=op_config)

# palettize weights
#compressed_mlmodel = cto.palettize_weights(model_from_torch, config)


#scripted_model = torch.jit.script(model.net)

#model_from_torch = coremltools.converters.convert(
#  scripted_model,
#  inputs=[ct.TensorType(name="input",
#                        shape=imgs.shape)])


#mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
#images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
#mimsave('example/out_2x.gif', images, fps=3)


print(f'=========================Done=========================')
