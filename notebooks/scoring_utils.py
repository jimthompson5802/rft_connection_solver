import numpy as np


def score_guess(guess, invalid_guesses, remaining_words, solution):
    """
    Scores a guess against solution groups.

    Args:
        guess: List of words that are being guessed to form a group
        remaining_words: List of words that are still in the game
        solution: Dictionary containing solution groups

    Returns:
        float: Score of the guess, or 0 if guess is invalid
    """
    # Check if guess is a subset of remaining_words
    if not set(guess).issubset(set(remaining_words)):
        return 0

    # Check if guess is already in invalid_guesses
    if any(set(guess) == set(invalid_group) for invalid_group in invalid_guesses):
        return 0

    # does guess have 4 words
    if len(set(guess)) != 4:
        return 0

    max_score = 0

    # Iterate through each solution group
    for group in solution["groups"]:
        score_for_group = 0
        solution_words = group["words"]

        # For each word in the guess that matches a word in the solution group, add 0.25
        for word in guess:
            if word in solution_words:
                score_for_group += 0.25

        # Update max_score if this group's score is higher
        max_score = max(max_score, score_for_group)

    return max_score


def compute_advantages(rewards: list):
    rewards = np.array(rewards)

    # Compute the mean and standard deviation of the rewards
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Avoid division by zero in case of zero variance (typically happens when all rewards are 0)
    # Note: In the GRPO implementation, we add 1e-4 to the std_reward to avoid division by zero
    if std_reward == 0:
        return [0] * len(rewards)

    # Divide by stddev of rewards to normalize range to 0
    advantages = (rewards - mean_reward) / std_reward
    return advantages.tolist()
