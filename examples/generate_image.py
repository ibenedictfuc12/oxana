from art_ai_agent_framework.generation.image_generation import ImageGenerator

def run_image_generation_example():
    generator = ImageGenerator()

    generator.train_dummy(epochs=1)

    illustration = generator.generate_illustration("Fantasy World", "Oil Painting")
    print("Generated Illustration:", illustration)

    product_design = generator.generate_product_design("Chair", "Minimalist")
    print("Generated Product Design:", product_design)

    nft_collection = generator.generate_nft_collection("MyNFT", 3)
    print("Generated NFT Collection:", nft_collection)

if __name__ == "__main__":
    run_image_generation_example()