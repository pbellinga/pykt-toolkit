from pykt.preprocess.data_proprocess import process_raw_data
from pykt.models.init_model import load_model

def test_process_raw_data():
    res = process_raw_data("assist2015", {"assist2015":"data/assist2015/2015_100_skill_builders_main_problems.csv"})
    print(res)

#test_process_raw_data()

def test_dkt():
    print("testing dkt")

    dataset = "data/assist2015/data.txt"

    # train_data, test_data = train_test_split(dataset, 0.2)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = load_model(model_name="dkt", model_config="configs/kt_config.json", data_config="assist2015", emb_type="", ckpt_path="")

# test_dkt()

from pykt.models.dkt import DKT

model = DKT(num_c=10, emb_size=10, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768)

print(model)

import torch
import lightning as pl
from torch.utils.data import DataLoader, TensorDataset

q = torch.randn(32, 10, 10)
r = torch.randn(32, 10, 10)

#train the model
class LitModel(pl.LightningModule):
    def __init__(self, model):
        super(LitModel, self).__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

# Dummy data
x = torch.randn(100, 10, 10)
y = torch.randint(0, 10, (100,))

dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the Lightning model
lit_model = LitModel(model)

# Train the model
trainer = pl.Trainer(max_epochs=10)
trainer.fit(lit_model, train_loader)

#test the model
