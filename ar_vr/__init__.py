from .interactive_installations import InteractiveInstallation

def init_ar_vr():
    return "AR/VR Module Ready"

class ARVRModule:
    def __init__(self):
        self.installation = InteractiveInstallation()

    def module_status(self):
        return init_ar_vr()