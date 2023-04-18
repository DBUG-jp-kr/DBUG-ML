from efficientnet_pytorch import EfficientNet

model_name = 'efficientnet-b0' 

image_size = EfficientNet.get_image_size(model_name)
model = EfficientNet.from_pretrained(model_name, num_classes=111)