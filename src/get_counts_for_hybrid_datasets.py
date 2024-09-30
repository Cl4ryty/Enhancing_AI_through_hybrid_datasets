import numpy as np
from collections import defaultdict

# Sample counts of real images for each class
real_counts = {1: 134, 2: 305, 3: 1315, 4: 60, 5: 156, 8: 41, 9: 109, 11: 34, 14: 177, 15: 116, 18: 355, 19: 1165, 20: 38, 21: 1375, 23: 290, 24: 344, 26: 80, 27: 66, 28: 37, 29: 40, 30: 38, 31: 59, 32: 53, 33: 122, 34: 132, 35: 102, 36: 47, 37: 66, 38: 33, 39: 46, 40: 72, 41: 35, 42: 62, 43: 65, 44: 102, 45: 142, 46: 77, 47: 42, 48: 167, 49: 101, 50: 50, 51: 82, 52: 71, 54: 94}

print("number of used classes", len(real_counts))
print("number of samples",  np.sum(list(real_counts.values())))

def calculate_sample_counts(real_counts, splits):
    real_data_counts = defaultdict(dict)
    synthetic_data_counts = defaultdict(dict)

    for real_ratio, synthetic_ratio in splits:
        total_ratio = real_ratio + synthetic_ratio

        for label, real_count in real_counts.items():
            # Calculate desired samples for real and synthetic
            desired_real_samples = int(real_count * (real_ratio / total_ratio))
            desired_synthetic_samples = int(
                real_count * (synthetic_ratio / total_ratio))

            # Correct the synthetic sample count to ensure the total matches
            combined_samples = desired_real_samples + desired_synthetic_samples

            # If the combined count does not match, adjust the counts
            if combined_samples < real_count:
                # Adjust synthetic samples to fill the gap
                needed = real_count - combined_samples
                desired_synthetic_samples += needed

            # Update the results in the respective dictionaries
            real_data_counts[(real_ratio, synthetic_ratio)][
                label] = desired_real_samples
            synthetic_data_counts[(real_ratio, synthetic_ratio)][
                label] = desired_synthetic_samples

    return dict(real_data_counts), dict(
        synthetic_data_counts)  # Convert defaultdict to regular dict


# Define the split ratios
split_ratios = [(0.1, 0.9), (0.5, 0.5)]  # (Real, Synthetic) pairs

# Calculate the sample counts for each split
real_data, synthetic_data = calculate_sample_counts(real_counts, split_ratios)

# Print the sample counts for real data in a single line format
real_data_str = "real_data_counts = {" + ", ".join([
                                                           f"'{real_ratio}/{synthetic_ratio}': \n{{{', '.join([f'{label}: {count}' for label, count in counts.items()])}}}\n"
                                                           for (real_ratio,
                                                                synthetic_ratio), counts
                                                           in
                                                           real_data.items()]) + "}"
print(real_data_str)

# Print the sample counts for synthetic data in a single line format
synthetic_data_str = "synthetic_data_counts = {" + ", ".join([
                                                                     f"'{real_ratio}/{synthetic_ratio}': \n{{{', '.join([f'{label}: {count}' for label, count in counts.items()])}}}\n"
                                                                     for (
                                                                     real_ratio,
                                                                     synthetic_ratio), counts
                                                                     in
                                                                     synthetic_data.items()]) + "}"
print(synthetic_data_str)
