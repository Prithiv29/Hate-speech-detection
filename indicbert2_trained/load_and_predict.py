# Reload the trained model and classify new comments
import json, torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
info = json.load(open("info.json"))
class EncoderMeanPool(nn.Module):
    def __init__(self, model_id):
        super().__init__(); self.enc = AutoModel.from_pretrained(model_id); h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.2); self.head = nn.Sequential(nn.Linear(h,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,2))
    def forward(self, input_ids, attention_mask):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        return self.head(self.drop((hs*m).sum(1)/m.sum(1).clamp(min=1e-9)))
tok = AutoTokenizer.from_pretrained(".")
model = EncoderMeanPool(info["model_id"]); model.load_state_dict(torch.load("model_state.pt")); model.float().eval()
def predict(texts):
    b = tok(texts, truncation=True, max_length=info["max_len"], padding=True, return_tensors="pt")
    with torch.no_grad(): return model(b["input_ids"], b["attention_mask"]).softmax(-1)[:, 1].tolist()   # P(hate)
print(predict(["example comment 1", "example comment 2"]))
