from .style_transfer import StyleTransferAgent

def init_style_transfer():
    return "Style Transfer Module Ready"

class StyleTransferModule:
    def __init__(self):
        self.agent = StyleTransferAgent()

    def module_status(self):
        return init_style_transfer()