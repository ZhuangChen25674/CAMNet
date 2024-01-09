import torch
x = torch.tensor([[1,2],[3,4]],dtype=torch.float,requires_grad=True)
min_value, min_index = torch.min(x,dim =1)
loss = torch.sum(min_value)
grad_x = torch.autograd.grad(loss, x)[0] 
print(min_value)
print(grad_x)
