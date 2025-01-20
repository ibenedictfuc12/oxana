import argparse
import time
from art_ai_agent_framework.style_transfer.style_transfer import StyleTransferAgent

def run_style_transfer_example(source_image, style_reference, video_mode=False):
    agent = StyleTransferAgent()
    if video_mode:
        output = agent.stylize_video(source_image, style_reference)
        print("Stylized Video:", output)
    else:
        output = agent.stylize_image(source_image, style_reference)
        print("Stylized Image:", output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="input_image.jpg")
    parser.add_argument("--style", default="VanGoghStyle")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    start = time.time()
    run_style_transfer_example(args.source, args.style, args.video)
    end = time.time()
    print("Elapsed:", end - start, "seconds")

if __name__ == "__main__":
    main()