def print_args(args):
    """Print experiment arguments in a simple format"""
    print('Args in experiment:')
    
    # Print args in a safer way to avoid encoding issues
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()
    
    # Special handling for zero-shot and few-shot
    if args.task_name == 'zero_shot_forecast':
        print(f"\033[1m[Zero-Shot]\033[0m Source: {args.data}, Target: {args.target_data}")
        print()
    elif args.task_name == 'few_shot_forecast':
        print(f"\033[1m[Few-Shot]\033[0m Data: {args.data}, Percent: {args.percent}")
        print()