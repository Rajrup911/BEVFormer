import torch
import torch.nn.functional as F

t1 = torch.load('bev_queries.pt')
t2 = torch.load('mlvl_feats.pt')
t3 = torch.load('prev_bev_old.pt')
t4 = torch.load('canbus.pt')
t5 = torch.load('prev_bev_rotated.pt')

print(t1.shape)
print(t2[0].shape)
print(t3.shape)
print(t5.shape)
print(t4)

#print(t1)
#print(t2)
#print(t3)

#mse = F.mse_loss(t1.cuda(), t2.cuda())
#print(mse)
