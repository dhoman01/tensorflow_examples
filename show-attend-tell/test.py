from model import Model
from utils import ArgumentParser, DataLoader
import matplotlib.pyplot as plt

args = ArgumentParser().parser.parse_args()
model = Model(args)

dl = DataLoader(args)
x,y = dl.next_batch()
img = x[0].reshape(28, 28 * 3);
plt.imshow(img)
plt.show()
print y[0]
