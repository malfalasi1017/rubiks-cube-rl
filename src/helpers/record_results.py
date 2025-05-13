import csv
import os

def save_results_to_csv(
    train_rewards,
    train_steps,
    test_rewards,
    test_steps,
    test_solved,
    hyperparams,
    output_path="results.csv"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate averages
    avg_train_rewards = sum(train_rewards) / len(train_rewards) if train_rewards else 0
    avg_train_steps = sum(train_steps) / len(train_steps) if train_steps else 0
    avg_test_rewards = sum(test_rewards) / len(test_rewards) if test_rewards else 0
    avg_test_steps = sum(test_steps) / len(test_steps) if test_steps else 0

    # Prepare row in the requested order
    row = [
        avg_train_rewards,
        avg_train_steps,
        avg_test_rewards,
        avg_test_steps,
        test_solved,
        hyperparams.get("SCRAMBLES"),
        hyperparams.get("MAX_STEPS"),
        hyperparams.get("BUFFER_SIZE"),
        hyperparams.get("BATCH_SIZE"),
        hyperparams.get("GAMMA"),
        hyperparams.get("LR"),
        hyperparams.get("EPS_START"),
        hyperparams.get("EPS_END"),
        hyperparams.get("EPS_DECAY"),
        hyperparams.get("TARGET_UPDATE_FREQ"),
        hyperparams.get("TRAIN_START"),
        hyperparams.get("NUM_EPISODES"),
    ]

    headers = [
        "avg train rewards",
        "avg of train steps",
        "avg test rewards",
        "avg test steps",
        "test_solved",
        "scrambles",
        "max_staps",
        "buffer_size",
        "batch_size",
        "gamma",
        "lr",
        "eps_start",
        "eps_end",
        "eps decay",
        "target_update_freq",
        "train_start",
        "num_episodes"
    ]

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"Results saved to {output_path}")