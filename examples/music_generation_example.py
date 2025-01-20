import argparse
import time
from art_ai_agent_framework.generation.music_generation import MusicGenerator

def run_music_generation_example(genre, length, epochs):
    music_gen = MusicGenerator()
    music_gen.train_dummy(epochs=epochs)
    track = music_gen.compose_track(genre, length)
    print("Composed Track:", track)
    sfx = music_gen.generate_sound_effects("Footstep", 8)
    print("Generated Sound Effect:", sfx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="Ambient")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    start_time = time.time()
    run_music_generation_example(args.genre, args.length, args.epochs)
    end_time = time.time()
    print("Elapsed:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()