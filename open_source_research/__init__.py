import argparse
import time
from art_ai_agent_framework.open_source_research.research_tools import ResearchTools

def run_open_source_research_example(args):
    research = ResearchTools()

    if args.action == "test_arch":
        start = time.time()
        result = research.test_new_architecture(args.architecture, args.dataset)
        end = time.time()
        print("Test New Architecture:", result)
        print("Elapsed:", end - start, "seconds")

    elif args.action == "compare_arch":
        start = time.time()
        result = research.compare_architectures(args.dataset)
        end = time.time()
        print("Compare Architectures:", result)
        print("Elapsed:", end - start, "seconds")

    elif args.action == "ablation":
        start = time.time()
        result = research.run_ablation_study(args.architecture, args.dataset, remove_layers=[False, True])
        end = time.time()
        print("Ablation Study Results:", result)
        print("Elapsed:", end - start, "seconds")

    elif args.action == "measure_perf":
        start = time.time()
        result = research.measure_performance(args.architecture, args.dataset, metric=args.metric)
        end = time.time()
        print("Measure Performance:", result)
        print("Elapsed:", end - start, "seconds")

    elif args.action == "benchmark_batch":
        start = time.time()
        batch_sizes = list(map(int, args.batch_sizes.split(","))) if args.batch_sizes else None
        result = research.benchmark_batch_sizes(args.architecture, args.dataset, batch_sizes=batch_sizes)
        end = time.time()
        print("Benchmark Batch Sizes:", result)
        print("Elapsed:", end - start, "seconds")

    elif args.action == "synthetic_exp":
        start = time.time()
        result = research.synthetic_experiment(args.architecture, steps=args.steps)
        end = time.time()
        print("Synthetic Experiment:", result)
        print("Elapsed:", end - start, "seconds")

    elif args.action == "resources":
        start = time.time()
        resources = research.provide_educational_resources()
        end = time.time()
        print("Educational Resources:", resources)
        print("Elapsed:", end - start, "seconds")

    else:
        print("No valid action specified. Use --help for details.")

def main():
    parser = argparse.ArgumentParser(
        description="Open Source Research Example using ResearchTools"
    )
    parser.add_argument("--action", type=str, default="test_arch",
                        help="Action to perform (test_arch, compare_arch, ablation, measure_perf, benchmark_batch, synthetic_exp, resources)")
    parser.add_argument("--architecture", type=str, default="resnet18",
                        help="Model architecture to use")
    parser.add_argument("--dataset", type=str, default="ArtDatasetV1",
                        help="Path or name of the dataset")
    parser.add_argument("--metric", type=str, default="loss",
                        help="Metric for measure_perf (loss or inference_time)")
    parser.add_argument("--batch_sizes", type=str, default="",
                        help="Comma-separated list of batch sizes for benchmark_batch")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of inference steps for synthetic_exp")
    args = parser.parse_args()

    run_open_source_research_example(args)

if __name__ == "__main__":
    main()