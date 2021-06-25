import os
import torch
ls = os.listdir("./black_model")
for i in ls:
    a=os.path.join("./black_model",i)
    b=torch.load(a)
    print(a)
    print(b["acc"])
