import argparse
import time
from art_ai_agent_framework.image_enhancement.image_enhancer import ImageEnhancer

def run_image_enhancement_example(file_path, scale_factor):
    enhancer = ImageEnhancer()
    noise_removed = enhancer.remove_noise(file_path)
    print(noise_removed)
    details_restored = enhancer.restore_details(file_path)
    print(details_restored)
    upscaled = enhancer.super_resolution(file_path, scale_factor)
    print(upscaled)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="test_image.jpg")
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    t0 = time.time()
    run_image_enhancement_example(args.file, args.scale)
    t1 = time.time()
    print("Elapsed:", t1 - t0, "seconds")

if __name__ == "__main__":
    main()