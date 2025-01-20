from .generation import init_generation_module
from .generation.image_generation import ImageGenerator
from .generation.music_generation import MusicGenerator
from .generation.text_generation import TextGenerator
from .style_transfer import init_style_transfer
from .style_transfer.style_transfer import StyleTransferAgent
from .image_enhancement import init_image_enhancement
from .image_enhancement.image_enhancer import ImageEnhancer
from .ar_vr import init_ar_vr
from .ar_vr.interactive_installations import InteractiveInstallation
# from .analysis import init_analysis
# from .analysis.classification import ArtAnalyzer
# from .education import init_education
# from .education.virtual_art_teacher import VirtualArtTeacher
# from .personalization import init_personalization
# from .personalization.personalization import ArtPersonalizer
# from .co_creation import init_co_creation
# from .co_creation.design_assistant import DesignAssistant
# from .marketing import init_marketing
# from .marketing.brand_tools import BrandStyler
from .open_source_research import init_open_source_research
from .open_source_research.research_tools import ResearchTools

def initialize_framework():
    return "Art AI Agent Framework Initialized"

def initialize_modules():
    statuses = []
    statuses.append(init_generation_module())
    statuses.append(init_style_transfer())
    statuses.append(init_image_enhancement())
    statuses.append(init_ar_vr())
    statuses.append(init_analysis())
    statuses.append(init_education())
    statuses.append(init_personalization())
    statuses.append(init_co_creation())
    statuses.append(init_marketing())
    statuses.append(init_open_source_research())
    return statuses

class ArtAIAgentFramework:
    def __init__(self):
        self.image_generator = ImageGenerator()
        self.music_generator = MusicGenerator()
        self.text_generator = TextGenerator()
        self.style_transfer_agent = StyleTransferAgent()
        self.image_enhancer = ImageEnhancer()
        self.interactive_installation = InteractiveInstallation()
        self.art_analyzer = ArtAnalyzer()
        self.virtual_art_teacher = VirtualArtTeacher()
        self.art_personalizer = ArtPersonalizer()
        self.design_assistant = DesignAssistant()
        self.brand_styler = BrandStyler()
        self.research_tools = ResearchTools()
        self.modules_status = initialize_modules()

    def framework_status(self):
        return {
            "framework": initialize_framework(),
            "modules": self.modules_status
        }