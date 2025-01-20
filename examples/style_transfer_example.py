from art_ai_agent_framework.style_transfer.style_transfer import StyleTransferAgent

def run_style_transfer_example():
    agent = StyleTransferAgent()

    stylized_image = agent.stylize_image("input_image.jpg", "VanGoghStyle")
    print("Stylized Image:", stylized_image)

    stylized_video = agent.stylize_video("input_video.mp4", "PicassoStyle")
    print("Stylized Video:", stylized_video)

if __name__ == "__main__":
    run_style_transfer_example()