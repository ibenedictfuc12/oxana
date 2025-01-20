from .image_enhancer import ImageEnhancer

def init_image_enhancement():
    return "Image Enhancement Module Ready"

class ImageEnhancementModule:
    def __init__(self):
        self.enhancer = ImageEnhancer()

    def module_status(self):
        return init_image_enhancement()