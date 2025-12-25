import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.tensor([[2,1000],[3,2000],[2,500],[1,800],[4,3000]], 
                      dtype=torch.float32,device=device)

labels = torch.tensor([[19],[31],[14],[15],[43]], dtype=torch.float32,device=device)

'''
最大归一化处理，将其化作
原始数据              归一化后
[2, 1000]     →     [0.5, 0.333...]
[3, 2000]     →     [0.75, 0.666...]
[2, 500]      →     [0.5, 0.166...]
[1, 800]      →     [0.25, 0.266...]
[4, 3000]     →     [1.0, 1.0]
这样可以保证每个特征的数据都在0-1之间，防止某个特征值过大对模型训练造成影响
'''
# inputs=inputs / torch.tensor([4,3000],dtype=torch.float32,device=device)

# 计算每个特征的均值和方差
mean = torch.mean(inputs, dim=0)
std = torch.std(inputs, dim=0)

# 标准化处理
inputs = (inputs - mean) / std
'''
这段代码初始化线性回归模型的参数：
w(权重):
形状 (2, 1):2 个输入特征对应 2 个权重
初始化为全 1
b(偏置):
形状 (1,)：标量偏置项
初始化为 1
'''
w=torch.ones(2,1,requires_grad=True,device=device)
b=torch.ones(1,requires_grad=True,device=device)

epoch = 2000
lr = 0.1

for i in range(epoch):
    outputs = inputs @ w + b
    # 计算平方差
    loss = torch.mean(torch.square(outputs - labels))

    print("loss:", loss.item())

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

new_inputs = torch.tensor([3,2500], dtype=torch.float32,device=device)

# 在进行预测前，也需要对新输入数据进行相同的标准化处理
new_inputs = (new_inputs - mean) / std
predicted = new_inputs @ w + b

# tolist() 将张量转换为列表，便于查看结果
print("Predicted value for input [3, 2500]:", predicted.tolist()[0][0])
