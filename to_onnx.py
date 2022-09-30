import torch
import torch.onnx
from model import SimpleNet

# A model class instance (class not shown)
model = SimpleNet()

# load the pth file
model.load_state_dict(torch.load('mnist_cnn.pt'))

sample_batch_size, channel, height, width = 1, 1, 28, 28

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(sample_batch_size, channel, height, width)

torch.onnx.export(model, dummy_input, "mnist_cnn.onnx")