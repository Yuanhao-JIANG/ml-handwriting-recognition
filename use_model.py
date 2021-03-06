import numpy
import glob
from PIL import Image
import torch
from torchvision import transforms
from model import LeNet5


def recognize(img_dir="./imgs", output_path="./output.txt", model_path="./saved_model/model.pt"):
    img_batch = []
    img_paths = glob.glob(img_dir+"/*")
    for i in range(len(img_paths)):
        img = Image.open(img_paths[i]).convert("L")
        x = transforms.ToTensor()(img)
        img_batch.append(x.numpy())

    # convert to numpy array first to make the following transformation to tensor faster
    img_batch = numpy.array(img_batch)
    img_batch = torch.tensor(img_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LeNet5(10)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = img_batch.to(device)
    with torch.no_grad():
        outputs, _ = model(x)
        outputs = outputs.detach().cpu()

        with open(output_path, "w") as f:
            for i in range(len(outputs)):
                f.write(f"{img_paths[i]} -> {torch.argmax(outputs[i]).item()}" + "\n")


# recognize()
