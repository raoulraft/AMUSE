import wandb
import csv
import random
import time

# Define max number of steps inside an episode 
n_max_steps = 100
# Define max number of episodes
n_max_episodes = 100


# Set the number of actions (0 = BPSK, 1 = 8PSK, 2 = 16PSK)
num_actions = 3

# Define the UCB1 exploration parameter
exploration_param = 2.0

# For each action, initialize the action-value estimates to 0 
action_counts = [0] * num_actions
total_rewards = [0] * num_actions


# Define wandb run parameters and start the wandb logging run
run_name = "ucb"
project_name = "DESERT"
wandb.init(name=run_name, project=project_name, tags=["static"], entity="unwis24", reinit=True)


# Function to select an action using UCB1
def select_action():
    for action in range(num_actions):
        if action_counts[action] == 0:
            return action  # Select unexplored action

    # Calculate ucb value for each arm, and return the arm with the max value
    ucb_values = [total_rewards[action] / action_counts[action] + exploration_param * math.sqrt(math.log(sum(action_counts)) / action_counts[action]) for action in range(num_actions)]
    return ucb_values.index(max(ucb_values))


internal_step = 1  # has to start from 1 to be able to make the desert sim interrupt every "interrupt_time" ms

# Define the episode counter
n_episode = 0

while(n_episode < n_max_episodes):

    # We set done = False, since done will be True when n_step > n_max_steps (defined later)
    with open('done.csv', 'w', newline='') as done_file:
        done_writer = csv.writer(done_file)
        done_writer.writerow([0])


    # Define the internal_step, used to synchronize this python script with DESERT
    # the CSVs will always have the internal step as the first number, followed by everything else (e.g, internal_step, action)
    internal_step = 1
    # state = np.zeros(input_shape)

    n_step = 1
    cumulative_reward = 0
    # Define the list to later calculate the mean throughput for each episode
    throughput_list = []

    while (n_step < n_max_steps):

        # Select an action based on the UCB1 algorithm
        action = select_action()

        # Write the chosen action to the actions CSV file
        with open('actions.csv', 'w', newline='') as actions_file:
            print("writing step and action actions.csv: {}, {}".format(internal_step, action))
            actions_writer = csv.writer(actions_file)
            actions_writer.writerow([internal_step, action])

        # Wait for the next reward to be written (by DESERT) to the file before taking the next step
        while True:
            with open('rewards.csv') as rewards_file_check:
                # format is like "2,,0.0, -2", we don't know why there are two commas :)

                rewards_reader_check = csv.reader(rewards_file_check)
                row = next(rewards_reader_check)
                step = int(row[0])
                reward = float(row[1])
                # state = float(row[2])
                print("[INFO] step and reward found in rewards.csv: {}, {}".format(step, reward))

            # Check if the two scripts are synchronized; if so, break
            if step == internal_step:
                cumulative_reward += reward
                # Insert throughput into the throughput list, used to calculate its average at the end of the episode
                throughput_list.append(reward)
                done = True if (n_step + 1) == n_max_steps else False
                print("[OK] step and reward found in rewards.csv: {}, {}".format(step, reward))
                break

            if step < internal_step:
                print("[WARNING] waiting for the reward to be written by AnPa (last_step < internal_step)")
                pass

            if step > internal_step:
                print(f"[WARNING] you messed up somewhere, since step found on csv ({step}) > internal_step ({internal_step})"
                      " (scripts are probably not coordinated on which step to start?)")
                pass

            time.sleep(0.2)

        # Log the step reward and step action in wandb
        wandb.log({"reward": reward},commit=False)
        wandb.log({"action": action},commit=False)
        # wandb.log({"state": state}, commit=True)

        # increase UCB1 counters and internal step counters
        action_counts[action] += 1
        total_rewards[action] += reward
        internal_step += 1
        n_step += 1

        # Check that both scripts are correctly going into the next step without missing any step
        while True:
            with open('synchronization.csv') as synchro_check:
                synchro_reader_check = csv.reader(synchro_check)
                row = next(synchro_reader_check)
                synchro_step = int(row[0])
                print(f"Read {synchro_step} in synchronization.csv, I was expecting {internal_step}")
                if synchro_step == internal_step or (n_step  == n_max_steps):
                    print("[OK] breaking from synchro check loop")
                    print(f"[INFO] step: {n_step}, n_max_steps: {n_max_steps}")
                    if (n_step == n_max_steps):
                        # Open the actions CSV file in write mode
                        with open('done.csv', 'w', newline='') as done_file:
                            done_writer = csv.writer(done_file)
                            # Write the action to the actions CSV file
                            done_writer.writerow([1])
                        time.sleep(5)
                    break
                time.sleep(1)

    # Calculate the mean throughput from the list defined earlier  
    mean_throughput = statistics.mean(throughput_list)

    # Log the mean throughput and cumulative reward (sum of rewards inside the episode) to wandb
    wandb.log({"mean throughput": mean_throughput, "episode": n_episode},commit=False)
    wandb.log({"cumulative reward": cumulative_reward, "episode": n_episode},commit=True)

    # Increase episode counter
    n_episode += 1


