import numpy as np
from PIL import Image
import functools

from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment
from model_tools.check_submission import check_models


from r3m import load_r3m

####preprocessing
def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() and 'P' not in pil_image.mode.upper():
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
           rgb_image = Image.new("RGB", pil_image.size)
           rgb_image.paste(pil_image)
           return rgb_image


def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
    transforms.Resize((image_size, image_size)),
    torchvision_preprocess(**kwargs),
    ])


def torchvision_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        lambda img: 255 * img.unsqueeze(0).to('cpu')
    ])


####actually define the model

def get_model_list():
    return ['r3m18']


def get_layers(name):
    assert name == 'r3m18'
    return [#'convnet.conv1',
            #'convnet.maxpool',
            #'convnet.layer1.1',
            'convnet.layer2.1',
            'convnet.layer3.1',
            'convnet.layer4.1',
            'convnet.fc']


preprocessing = functools.partial(load_preprocess_images, image_size=224)

r3m18cpu = load_r3m("resnet18") 
r3m18cpu.eval();
r3m18cpu = r3m18cpu.module.to('cpu')
activations_model = PytorchWrapper(identifier='r3m18',
                                   model=r3m18cpu,
                                   preprocessing=preprocessing)

model = ModelCommitment(identifier='r3m18',
                        activations_model=activations_model,
                        layers = get_layers('r3m18'))


def get_model(name):
    assert name == 'r3m18'
    r3m18_wrapper = activations_model
    r3m18_wrapper.image_size = 224
    return r3m18_wrapper


def get_bibtex(model_identifier):
    return """@misc{2203.12601,
                    Author = {Suraj Nair and Aravind Rajeswaran and Vikash Kumar and Chelsea Finn and Abhinav Gupta},
                    Title = {R3M: A Universal Visual Representation for Robot Manipulation},
                    Year = {2022},
                    Eprint = {arXiv:2203.12601}
                   }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

