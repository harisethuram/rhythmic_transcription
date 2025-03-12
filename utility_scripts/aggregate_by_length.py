import json
import os

if __name__ == "__main__":
    filter_dir = "test_results/URMP_hybrid/"
    subdirs_filter = [f for f in os.listdir(filter_dir) if os.path.isdir(os.path.join(filter_dir, f))]
    
    main_dir = "test_results/URMP/"
    
    total_filter_error = 0
    total_main_error = 0
    
    for subdir in subdirs_filter:
        main_results = json.load(open(os.path.join(main_dir, subdir, "results.json"), "r"))
        filter_results = json.load(open(os.path.join(filter_dir, subdir, "results.json"), "r"))
        
        filter_error = filter_results["error"]["mean_onset_lengths_diff"]
        main_error = main_results["error"]["mean_onset_lengths_diff"]
        
        total_filter_error += filter_error
        total_main_error += main_error
        
    mean_filter_error = total_filter_error / len(subdirs_filter)
    mean_main_error = total_main_error / len(subdirs_filter)
    
    print(f"Mean filter error: {mean_filter_error}")
    print(f"Mean main error: {mean_main_error}")
        
        