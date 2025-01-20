from .image_generation import ImageGenerator
from .music_generation import MusicGenerator
from .text_generation import TextGenerator

def init_generation_module():
    return "Generation Module Ready"

class GenerationModule:
    def __init__(self):
        self.image_generator = ImageGenerator()
        self.music_generator = MusicGenerator()
        self.text_generator = TextGenerator()

    def module_status(self):
        return init_generation_module()