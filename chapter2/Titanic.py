from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd

class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # 定义数据的均值和标准差
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910,
            "Pclass_Pclass": 5.704482,
            "Pclass_Age": 61.938151,
            "Pclass_SibSp": 1.198880,
            "Pclass_Parch": 0.983193,
            "Pclass_Fare": 53.052327,
            "Pclass_Sex_female": 0.754902,
            "Age_Age": 1092.761169,
            "Age_SibSp": 11.066415,
            "Age_Parch": 10.470476,
            "Age_Fare": 1104.142053,
            "Age_Sex_female": 10.204482,
            "SibSp_SibSp": 1.126050,
            "SibSp_Parch": 0.525210,
            "SibSp_Fare": 24.581262,
            "SibSp_Sex_female": 0.233894,
            "Parch_Parch": 0.913165,
            "Parch_Fare": 24.215465,
            "Parch_Sex_female": 0.259104,
            "Fare_Fare": 4000.200255,
            "Fare_Sex_female": 17.393698,
            "Sex_female_Sex_female": 0.365546
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274,
            "Pclass_Pclass": 3.447593,
            "Pclass_Age": 34.379609,
            "Pclass_SibSp": 2.603741,
            "Pclass_Parch": 2.236945,
            "Pclass_Fare": 52.407209,
            "Pclass_Sex_female": 1.118572,
            "Age_Age": 991.079188,
            "Age_SibSp": 19.093099,
            "Age_Parch": 29.164503,
            "Age_Fare": 1949.356185,
            "Age_Sex_female": 15.924481,
            "SibSp_SibSp": 3.428831,
            "SibSp_Parch": 1.561298,
            "SibSp_Fare": 70.185369,
            "SibSp_Sex_female": 0.639885,
            "Parch_Parch": 3.008314,
            "Parch_Fare": 77.207321,
            "Parch_Sex_female": 0.729143,
            "Fare_Fare": 19105.110593,
            "Fare_Sex_female": 43.568303,
            "Sex_female_Sex_female": 0.481921
        }

        self.data = self._load_data()
        # 去掉survived这个标签列，计算特征数量
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df = df.dropna(subset=["Age"])
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female"]

        for i in range(len(base_features)):
            for j in range(i, len(base_features)):
                df[base_features[i] + "_" + base_features[j]] = ((df[base_features[i]] * df[base_features[j]]
                                                                  - self.mean[
                                                                      base_features[i] + "_" + base_features[j]])
                                                                 / self.std[base_features[i] + "_" + base_features[j]])
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 取idx行对应的数据，分别取出特征和标签
        # 取features
        features = self.data.drop(columns=["Survived"]).iloc[idx].values

        # 取labels
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        return out

train_dataset = TitanicDataset(r"./data/train.csv")

model = LogisticRegressionModel(train_dataset.feature_size)
model.to("cuda")

# 将模型设置为训练模式
model.train()

# 用优化器来管理参数，并用它管理学习器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    correct = 0
    step = 0
    total_loss = 0

    # 这里的batch_size是256，所以数据大小是256
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        features = features.to("cuda")
        labels = labels.to("cuda")

        # 清空上一步的梯度
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))

        # 交叉熵损失函数
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()

        # 计算参数梯度
        loss.backward()

        # 对参数进行更新
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {total_loss/step:.4f}')
    print(f'Training Accuracy: {correct / len(train_dataset)}')