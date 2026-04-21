from src.models.baseline import BaselineModel
from src.models.cnn_model import CNNModel

# choose model
model_type = "cnn"  # or "baseline"

if model_type == "baseline":
    model = BaselineModel(vocab_size=10000, embed_dim=100, num_classes=6)
else:
    model = CNNModel(vocab_size=10000, embed_dim=100, num_classes=6)

print(model)