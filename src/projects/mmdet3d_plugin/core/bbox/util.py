import torch 

def onnx_atan2(y, x):
    # Create a pi tensor with the same device and data type as y
    pi = torch.tensor(torch.pi, device=y.device, dtype=y.dtype)
    half_pi = pi / 2
    eps = 1e-6

    # Compute the arctangent of y/x
    ans = torch.atan(y / (x + eps))

    # Create boolean tensors representing positive, negative, and zero values of y and x
    y_positive = y > 0
    y_negative = y < 0
    x_negative = x < 0
    x_zero = x == 0

    # Adjust ans based on the positive, negative, and zero values of y and x
    ans += torch.where(y_positive & x_negative, pi, torch.zeros_like(ans))  # Quadrants I and II
    ans -= torch.where(y_negative & x_negative, pi, torch.zeros_like(ans))  # Quadrants III and IV
    ans = torch.where(y_positive & x_zero, half_pi, ans)  # Positive y-axis
    ans = torch.where(y_negative & x_zero, -half_pi, ans) 

def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes