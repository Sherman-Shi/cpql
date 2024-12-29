import torch as th
import numpy as np
import math
from scipy.special import erfinv
from scipy.stats import qmc


def Normdf_inv(sample):
    """
    Convert uniform samples to standard normal samples using inverse CDF (erfinv).
    """
    v = 0.5 + (1 - np.finfo(sample.dtype).eps) * (sample - 0.5)
    norm_sample = erfinv(2 * v - 1) * math.sqrt(2)
    return norm_sample


def generate_qmc_normal_samples(dim, num_steps_per_epoch, sample_num, seed=None):
    """
    Generate (num_steps_per_epoch, sample_num) groups of independent QMC samples.
    
    Each group contains 'sample_num' samples drawn from a multivariate normal distribution.

    Parameters:
    - dim (int): Dimension of the normal distribution (number of independent variables).
    - num_steps_per_epoch (int): Number of groups or steps in one epoch.
    - sample_num (int): Number of samples per group.
    - seed (int, optional): Seed for random number generator used in scrambling.

    Returns:
    - Tensor: Shape (num_steps_per_epoch, sample_num, dim)
    """
    # Initialize QMC sampler
    qmc_sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    all_samples = []

    # Generate QMC samples for each group independently
    for step in range(num_steps_per_epoch):
        # Generate uniform QMC samples
        uniform_samples = qmc_sampler.random(n=sample_num)

        # Convert to normal distribution using inverse CDF
        normal_samples = Normdf_inv(uniform_samples)

        # Append samples for this step
        all_samples.append(normal_samples)

        # Re-seed to make groups independent
        qmc_sampler = qmc.Sobol(d=dim, scramble=True, seed=(seed + step + 1) if seed is not None else None)

    # Convert list to tensor of shape (num_steps_per_epoch, sample_num, dim)
    return th.tensor(np.array(all_samples), dtype=th.float32)


class SampledAgent:
    def __init__(self, action_dim, num_steps_per_epoch, sample_num, seed):
        self.action_dim = action_dim
        self.num_steps_per_epoch = num_steps_per_epoch
        self.sample_num = sample_num
        self.seed = seed
        self.normal_samples = None
        self.sample_index = [0] * num_steps_per_epoch  # Track sample indices per step
        self.used_samples = [set() for _ in range(num_steps_per_epoch)]  # Track used samples for each step
        np.random.seed(seed)
        th.manual_seed(seed)

    def generate_qmc_samples(self):
        """
        Generate (num_steps_per_epoch, sample_num) independent QMC samples.
        """
        self.normal_samples = generate_qmc_normal_samples(
            dim=self.action_dim,
            num_steps_per_epoch=self.num_steps_per_epoch,
            sample_num=self.sample_num,
            seed=self.seed
        )
        self.sample_index = [0] * self.num_steps_per_epoch  # Reset the sample index
        self.used_samples = [set() for _ in range(self.num_steps_per_epoch)]  # Reset used samples
        print("Generated new QMC normal samples.")

    def get_normal_sample(self, step):
        """
        Retrieve one unique QMC sample for a given timestep (step) without replacement.

        If all samples have been used, regenerate the QMC samples. If all samples are used
        in a step, the set of used samples is cleared to allow reuse in future episodes.
        """
        if self.normal_samples is None or step >= self.num_steps_per_epoch:
            # Regenerate samples if none exist or step exceeds bounds
            self.generate_qmc_samples()

        # Ensure that the sample index is within the range of available samples
        available_samples = set(range(self.sample_num)) - self.used_samples[step]

        if len(available_samples) == 0:
            # If all samples have been used, clear the used samples and regenerate the samples
            print(f"All samples used for step {step}. Clearing used samples and regenerating QMC samples.")
            self.used_samples[step] = set()  # Clear used samples
            self.generate_qmc_samples()
            available_samples = set(range(self.sample_num))  # Reset available samples
        
        # Select one sample randomly from available samples
        selected_sample = np.random.choice(list(available_samples))

        # Mark this sample as used for this step
        self.used_samples[step].add(selected_sample)

        # Return the selected sample for this step
        return self.normal_samples[step][selected_sample]
