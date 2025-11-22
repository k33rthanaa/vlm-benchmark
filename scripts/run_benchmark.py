#!/usr/bin/env python3
import argparse
import sys
import os
import yaml

# Add package root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlm_benchmark.evaluate import Evaluator, run_benchmark, BenchmarkConfig

def load_config(config_path: str) -> BenchmarkConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return BenchmarkConfig(**config_dict)

def main():
    parser = argparse.ArgumentParser(description="Run VLM Benchmark")
    
    # Config argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Legacy arguments (backward compatibility)
    parser.add_argument("--model_type", type=str, help="Model type (e.g., clip)")
    parser.add_argument("--model_name", type=str, help="HuggingFace model name")
    parser.add_argument("--task", type=str, help="Task to run (e.g., retrieval)")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")

    args = parser.parse_args()

    # Mode 1: Config file
    if args.config:
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        
        print(f"Loading config from {args.config}")
        try:
            config = load_config(args.config)
            run_benchmark(config)
            sys.exit(0)
        except Exception as e:
            print(f"Error running benchmark from config: {e}")
            sys.exit(1)

    # Mode 2: CLI arguments
    if args.model_type and args.model_name and args.task and args.dataset:
        print(f"Starting benchmark with CLI args:")
        print(f"  Model: {args.model_type} ({args.model_name})")
        print(f"  Task: {args.task}")
        print(f"  Dataset: {args.dataset}")

        try:
            evaluator = Evaluator(args.model_type, args.model_name, args.task, args.dataset)
            results = evaluator.evaluate()
            print("Evaluation Results:")
            print(results)
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            sys.exit(1)
    
    # Mode 3: Default config
    default_config_path = os.path.join("configs", "default.yaml")
    if os.path.exists(default_config_path):
        print(f"No arguments provided. Using default config: {default_config_path}")
        try:
            config = load_config(default_config_path)
            run_benchmark(config)
            sys.exit(0)
        except Exception as e:
            print(f"Error running benchmark from default config: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
