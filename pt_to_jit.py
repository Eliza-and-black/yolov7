import yaml
import torch
import argparse
import numpy as np
from models.yolo import Model
from utils.torch_utils import intersect_dicts
import torch.nn as nn
from models.common import Conv

height, width = 416, 768  # 注意此处导出宽高尺寸 == 生成ldmb尺寸 == prototxt前处理尺寸

parser = argparse.ArgumentParser()
# 重要的事情：一定要用cfg/training/yolov7x.yaml，不能用cfg/deploy/yolov7x.yaml,否则会因为和训练时的网络结构不一致，导致预测结果异常.
parser.add_argument('--cfg', type=str, default='cfg/training/yolov7x.yaml', help='model.yaml path')
parser.add_argument('--input_model', type=str, default='weights/Animal_x-d-416-768_20230510.pt', help='input pt weights path')
parser.add_argument('--save_model', type=str, default='weights/Animal_x-d-416-768_20230510_jit.pt', help='save jit weights path')
opt = parser.parse_args()


if isinstance(opt.cfg, str):
    with open(opt.cfg, errors='ignore') as f:
        cfg_content = yaml.safe_load(f)  # load cfg dict
nc = cfg_content.get('nc')
print(nc)

ckpt = torch.load(opt.input_model, map_location='cpu')  # load checkpoint
model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device='cpu')  # create
state_dict = ckpt['model'].float().state_dict()  # to FP32
state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
model.load_state_dict(state_dict, strict=False)  # load

model.eval()

x = torch.zeros(1, 3, height, width)  # 注意此处导出宽高尺寸 == 生成ldmb尺寸 == prototxt前处理尺寸

# 导出jitmodel必须先dry run一次，否则会报错
with torch.no_grad():
    output1 = model(x)  # dry run

jitmodel = torch.jit.trace(model, x, strict=False)

with torch.no_grad():
    output2 = jitmodel(x)

#对比模型输出结果
print('output1[0].shape', output1[0].shape)
print('output2[0].shape', output2[0].shape)

# print('output1', output1)
# print('output2', output2)
np.testing.assert_almost_equal(output1[0].numpy(), output2[0].numpy(), decimal=4)

torch.jit.save(jitmodel, opt.save_model)
print("->>模型转换成功！")




