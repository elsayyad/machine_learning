import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models.densenet import model_urls
from collections import OrderedDict
import numpy as np
import json
from PIL import Image
import argparse

def load_model(filepath, device = "cuda"):
    if device == "cuda" and torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        print("WARNING! - switching to cpu .. No GPU found!")

    model_checkpoint = torch.load(filepath, map_location=map_location)

    model_base_name = model_checkpoint["model_base_name"]
    model_input = model_checkpoint["model_input"]
    model_hidden_layer = model_checkpoint["model_hidden_layer"]

    #model = models.densenet121(pretrained=False)
    model = eval("models."+model_base_name+"(pretrained=False)")
    for param in model.parameters():
        param.requires_grad = False


    classifier  = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model_input, model_hidden_layer)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(model_hidden_layer, 102)),
        ('dropout', nn.Dropout(p=model_checkpoint['dropout'])),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier

    model.load_state_dict(model_checkpoint['model_state_dict'])
    optimizer = model_checkpoint['optimizer']
    optimizer.load_state_dict(model_checkpoint['optim_state_dict'])
    model.class_to_idx = model_checkpoint['model_class_to_idx']

    return model

def process_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(225),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()
                                   ])
    pil_img = Image.open(image)
    pil_img_mod = transform(pil_img).float()

    np_pil_img = np.array(pil_img_mod)
    np_pil_img = (np.transpose(np_pil_img, (1, 2, 0)) - mean)/std
    np_pil_img_trans = np.transpose(np_pil_img, (2, 0, 1))

    return torch.from_numpy(np_pil_img_trans)

def predict(image_path, checkpoint, topk, device, cat_names_file = None):
#def predict(image_path, checkpoint, topk=topk, cat_names_file=None, device):
    model = load_model(checkpoint, device = device)

    if device == "cuda" and torch.cuda.is_available():
            image_processed = process_image(image_path).unsqueeze(0).type(torch.cuda.FloatTensor)
    else:
            image_processed = process_image(image_path).unsqueeze(0).type(torch.FloatTensor)

    # TODO: Implement the code to predict the class from an image file
    print("device used is .. ", device)

    image_processed.to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        logps = model(image_processed)
        logps = logps.to(torch.device('cpu'))

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

        top_p = top_p.numpy()[0]            ## Top probability
        top_class = top_class.numpy()[0]    ## Top class number

        idx_to_class = {val: key for key, val in model.class_to_idx.items()}    ## Class Number to Name

        top_class = [idx_to_class[i] for i in top_class]    ## Top class numbers to names conversion

        flower_class_names = "Empty"
        if cat_names_file is not None:
            with open(cat_names_file , 'r') as f:
                cat_to_name = json.load(f)
            flower_class_names = [cat_to_name[str(c)] for c in top_class]

            actual_input_class_no = int(image_path.split('/')[3])
            actual_input_class_name = flower_class_names[actual_input_class_no]


    return top_p, top_class, flower_class_names, actual_input_class_name

def main():
    parser = argparse.ArgumentParser(description="Predicting from a Nureal Network Model")
    parser.add_argument('input', type=str, help="Path for image to be predicted. Path should be <dir>/<dataset dir>/<class>/<image file path>. i.e ./flowers/test/1/image_06743.jpg")
    parser.add_argument('checkpoint', type=str, help='Checkpoint of trained model to predict from')
    parser.add_argument('--category_names', type=str, help="Use a mapping of categories to real names")
    parser.add_argument('--topk', type=int, help="A number of most likely k classes")
    parser.add_argument('--gpu', action='store_true', help="Use GPU (cuda) if available otherwise CPU")

    args, _ = parser.parse_known_args()

    image_path = args.input
    checkpoint = args.checkpoint

    topk = 1
    if args.topk:
        topk = args.topk

    category_names = None
    if args.category_names:
        category_names = args.category_names

    device = "cpu"
    if args.gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    #print(process_image(image_path))
    probs, predicted_classes, class_names, actual_input_class_name = predict(image_path, checkpoint, topk=topk,  device=device, cat_names_file=category_names)


    print("Predicted class(es) names: {}".format(class_names))
    print("Predicted Probabilities: {}".format(probs))
    print("Predicted class(es) number(s): {}".format(predicted_classes))
    print("Actual image input: {}".format(actual_input_class_name))


if __name__ == '__main__':
    main()
