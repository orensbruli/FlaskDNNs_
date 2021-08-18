from PIL import Image
from torchvision import transforms
import urllib
import torch

repos = {'pytorch/vision:v0.10.0': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', "alexnet",
                                    'vgg11',
                                    'vgg11_bn',
                                    'vgg13',
                                    'vgg13_bn',
                                    'vgg16',
                                    'vgg16_bn',
                                    'vgg19',
                                    'vgg19_bn',
                                    'googlenet',
                                    'shufflenet_v2_x1_0']}


def get_tags_for_image(filename, repo_name="pytorch/vision:v0.10.0", model_name="googlenet"):

    model = torch.hub.load(repo_name, model_name, pretrained=True)
    model.eval()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    with open("resources/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 10)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

if __name__ == '__main__':
    repo_name = list(repos.keys())[0]
    model_name = repos[repo_name][0]
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    url, filename = (
    "https://images.freeimages.com/images/small-previews/199/sunflowers-6-1392951.jpg", "blank-sign-bullet-holes.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    get_tags_for_image(filename, repo_name, model_name)