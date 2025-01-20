import argparse
import time
from art_ai_agent_framework.generation.image_generation import ImageGenerator

def run_image_generation_example(epochs):
    start_time = time.time()
    generator = ImageGenerator()
    generator.train_dummy(epochs=epochs)
    illustration = generator.generate_illustration("Fantasy World", "Oil Painting")
    print("Generated Illustration:", illustration)
    product_design = generator.generate_product_design("Chair", "Minimalist")
    print("Generated Product Design:", product_design)
    nft_collection = generator.generate_nft_collection("MyNFT", 3)
    print("Generated NFT Collection:", nft_collection)
    elapsed = time.time() - start_time
    print("Elapsed:", elapsed, "seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    run_image_generation_example(epochs=args.epochs)

if __name__ == "__main__":
    main()