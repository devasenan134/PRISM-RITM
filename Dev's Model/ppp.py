import torch
import time
print(torch.version)
print(torch.cuda.is_available())

a = torch.randn(10000,1000)
b = torch.randn(1000,2000)

device = torch.device('cuda:0')
a = a.to(device)
b = b.to(device)

start_time = time.time()
c = torch.matmul(a,b)
end_time2 = time.time()
print(a.device,end_time2-start_time,c.norm(2))