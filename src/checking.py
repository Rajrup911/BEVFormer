import onnx
import onnxsim
import onnxruntime as ort
import torch
import numpy as np
#torch.set_printoptions(1000000)

model_onnx_2 = onnx.load("/home/ava/rajrup/new_bev/BEVFormer/bfv_prev_bev.onnx")
model_onnx_2, check = onnxsim.simplify(model_onnx_2, check_n=3)
assert check, 'assert check failed'
onnx.save(model_onnx_2, 'bfv_prev_bev-sim.onnx')

model_onnx_1 = onnx.load("/home/ava/rajrup/new_bev/BEVFormer/bfv_wo_prev_bev.onnx")
model_onnx_1, check = onnxsim.simplify(model_onnx_1, check_n=3)
assert check, 'assert check failed'
onnx.save(model_onnx_1, 'bfv_wo_prev_bev-sim.onnx')
'''
prev_bev_old = torch.load('prev_bev_old.pt')
canbus = torch.load('canbus.pt')
prev_bev_rotated = torch.load('prev_bev_rotated.pt')
#print(canbus)

sess = ort.InferenceSession("/home/ava/rajrup/new_bev/BEVFormer/transformerv2.onnx")
inputs = {'canbus': canbus[0].astype(np.float32), 'prev_bev': prev_bev_old.cpu().numpy()}
output = sess.run(None, inputs)
x = torch.Tensor(output[0])
#print(output)
print(canbus[0][-1])
print(x)
print(prev_bev_rotated)

import torch.nn.functional as F
mse = F.mse_loss(x.cuda(), prev_bev_rotated.cuda())
print(mse)
'''
