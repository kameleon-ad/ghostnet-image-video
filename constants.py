from torchvision.transforms import transforms


INPUT_SIZE = (256, 256, )
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
])

__all__ = [
    "INPUT_SIZE",
    "RGB_MEAN",
    "RGB_STD",
    "TRANSFORM"
]
