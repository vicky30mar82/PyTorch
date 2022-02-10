import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
X, y = datasets.make_blobs(n_samples = n_pts, random_state = 123, centers = centers, cluster_std = 0.4)
x_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(100,1))

def scatter_plot():
  plt.scatter(X[y == 0, 0], X[y == 0, 1])
  plt.scatter(X[y == 1, 0], X[y == 1, 1])
scatter_plot()

class Model(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = torch.sigmoid(self.linear(x))
    return pred
  def predict(self, x):
    pred = self.forward(x)
    if pred >= 0.5:
      return 1
    else:
      return 0


torch.manual_seed(2)
model = Model(2,1)

[w,b] = model.parameters()
w1, w2 = w.view(2)
b1 = b[0]
def get_params():
  return (w1.item(), w2.item(), b[0].item())

def plot_fit(title):
  plt.title = title
  w1, w2, b1 = get_params()
  x1 = np.array([-2.0, 2.0])
  x2 = ((w1 * x1) + b1)/(-w2)
  plt.plot(x1, x2, 'r')
  scatter_plot()
	
plot_fit("init")

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 1000
losses = []
for i in range(epochs):
  y_pred = model.forward(x_data)
  loss = criterion(y_pred, y_data)
  print("Epoch:", i , "Loss: ", loss.item())
  losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
	
plt.plot(range(epochs), losses)

pt1 = torch.Tensor([1.0, -1.0])
pt2 = torch.Tensor([-1.0, 1.0])
plt.plot(pt1.numpy()[0], pt1.numpy()[1], 'ro')
plt.plot(pt2.numpy()[0], pt2.numpy()[1], 'ko')

print("Red: ", format(model.forward(pt1).item()))
print("Black: ", format(model.forward(pt2).item()))

print("Red: ", format(model.predict(pt1)))
print("Black: ", format(model.predict(pt2)))
plot_fit('Trained Model')
