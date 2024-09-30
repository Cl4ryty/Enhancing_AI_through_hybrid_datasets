import random
import sys
from copy import deepcopy

import numpy as np


def calculate_synthetic_counts(original_counts, desired_synthetic_percentage):
    total_real_samples = sum(original_counts.values())
    print("total samples in original dataset", total_real_samples)
    classes_in_dataset = len(original_counts)
    samples_per_class = total_real_samples // classes_in_dataset
    extra_samples = total_real_samples - samples_per_class*classes_in_dataset
    total_number_of_synthetic_samples = round(total_real_samples * desired_synthetic_percentage)
    print("samples per class", samples_per_class)
    print("extra_samples", extra_samples)
    class_labels = original_counts.keys()
    # get samples per class
    samples_for_classes = [samples_per_class] * classes_in_dataset

    missing_real = np.array(samples_for_classes)-np.array(list(original_counts.values()))
    missing_real[missing_real<0] = 0
    print("missing real", dict(zip(class_labels, missing_real)))
    print("total missing real count", np.sum(missing_real),"percentage of data", np.sum(missing_real)/total_real_samples)


    samples_for_classes = dict(zip(class_labels, samples_for_classes))


    synthetic_counts = dict(zip(class_labels, [0]*classes_in_dataset))
    print("synthetic counts", synthetic_counts)
    real_counts = dict([[label, count] if count<samples_per_class else [label, samples_per_class] for label, count in original_counts.items()])
    print("real_counts", real_counts)


    # fill up classes with fewer samples than the required sample count per class
    for label, real_count in real_counts.items():
        if real_count < samples_per_class:
            # add synthetic counts
            synthetic_counts[label] = samples_per_class-real_count


    print("synthetic counts", synthetic_counts)

    # figure out if this already reaches the set percentages
    number_of_synthetic = sum(synthetic_counts.values())
    if number_of_synthetic/total_real_samples > desired_synthetic_percentage:
        print("!!warning: creating a balanced dataset will lead to a larger synthetic percentage that requested!!!", file=sys.stderr)

    # otherwise, keep filling up other classes equally
    else:
        # get the number of synthetic samples still needed to get to the specified percentage
        missing_synthetic_samples = total_number_of_synthetic_samples - number_of_synthetic
        print("missing synthetic samples", missing_synthetic_samples)

        # create bag of classes to select for extra samples
        classes_to_select_from = list(original_counts.keys())
        classes_with_real_values_to_select_from = [key for key, value in original_counts.items() if value > samples_per_class]

        # keep going until the desired percentage of synthetic samples is reached
        while missing_synthetic_samples > 0:

            # print("\nsynthetic counts", synthetic_counts)
            # print("real_counts", real_counts)

            # randomly select class with extra counts
            while extra_samples > 0 and missing_synthetic_samples > 0:
                print("extra samples 1")
                class_choice = random.choice(classes_to_select_from)
                # remove choice from list so that it cannot be chosen again
                classes_to_select_from.remove(class_choice)
                samples_for_classes[class_choice] += 1
                extra_samples -= 1
                # add synthetic sample if there are no real ones  left
                if original_counts[class_choice] < samples_for_classes[class_choice]:
                    missing_synthetic_samples -= 1
                    synthetic_counts[class_choice] += 1
                else:
                    # add real one
                    real_counts[class_choice] += 1

            # replace real samples in class with fewest synthetic samples
            s_counts_array = np.array(list(synthetic_counts.values()))
            index = np.argmin(s_counts_array)
            label = list(synthetic_counts.keys())[index]

            # print("index", index, "label", label)
            real_counts[label] -= 1
            synthetic_counts[label] += 1
            missing_synthetic_samples -= 1

        while extra_samples:
            # select only from classes where real samples can be used

            # create new bag only if it runs out of classes to choose from
            if len(classes_with_real_values_to_select_from)==0:
                classes_with_real_values_to_select_from = [key for key, value in original_counts.items() if value > samples_per_class]

            class_choice = random.choice(classes_with_real_values_to_select_from)
            classes_with_real_values_to_select_from.remove(class_choice)
            samples_for_classes[class_choice] += 1
            real_counts[class_choice] += 1
            extra_samples -= 1


    return synthetic_counts, real_counts


# Example usage:
original_counts = {1: 134, 2: 305, 3: 1315, 4: 60, 5: 156, 8: 41, 9: 109, 11: 34, 14: 177, 15: 116, 18: 355, 19: 1165, 20: 38, 21: 1375, 23: 290, 24: 344, 26: 80, 27: 66, 28: 37, 29: 40, 30: 38, 31: 59, 32: 53, 33: 122, 34: 132, 35: 102, 36: 47, 37: 66, 38: 33, 39: 46, 40: 72, 41: 35, 42: 62, 43: 65, 44: 102, 45: 142, 46: 77, 47: 42, 48: 167, 49: 101, 50: 50, 51: 82, 52: 71, 54: 94}

# Desired synthetic percentage
desired_synthetic_percentage = 1.0  # For instance, 40% synthetic samples

# Calculate synthetic counts
synthetic_counts, real_counts = calculate_synthetic_counts(original_counts,
                                                        desired_synthetic_percentage)

print("\n\n")
# Print results
print("Synthetic Counts to Generate:")
print(synthetic_counts)

print("real counts")
print(real_counts)

combined_counts = deepcopy(real_counts)
for label, value in synthetic_counts.items():
    combined_counts[label] += value
print("combined counts", combined_counts)

print("number of real samples", sum(real_counts.values()),"number of synthetic samples", sum(synthetic_counts.values()))
print("total number of samples", sum(combined_counts.values()), "original total", sum(original_counts.values()))
print("percentage of synthetic samples", sum(synthetic_counts.values())/sum(combined_counts.values()), "requests percentage", desired_synthetic_percentage)

