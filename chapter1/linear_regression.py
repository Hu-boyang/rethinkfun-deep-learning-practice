import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成一个 100x3 的随机张量
inputs=torch.rand(100,3)

weights=torch.tensor([[1.1],[2.2],[3.3]])

bias=torch.tensor(4.4)

# 可以用matmul也可以用@表示矩阵乘法
targets=inputs @ weights + bias + 0.1*torch.randn(100,1)

w=torch.rand((3,1),requires_grad=True,device=device)
b=torch.rand((1,),requires_grad=True,device=device)

inputs=inputs.to(device)
targets=targets.to(device)
epoch=10000
lr=0.003

for i in range(epoch):
    outputs=inputs @ w + b
    loss=torch.mean(torch.square(outputs - targets))
    print("loss:", loss.item())
    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

    
    w.grad.zero_()
    b.grad.zero_()

print("Trained weights:", w)
print("Trained bias:", b)
