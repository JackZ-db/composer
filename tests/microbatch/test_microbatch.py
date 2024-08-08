import torch
from torch.utils.data import DataLoader
from composer import Trainer
from composer.models import ComposerModel
from tests.common import SimpleModel, RandomClassificationDataset
from composer.utils import dist, reproducibility
import pytest 

@pytest.mark.gpu
def test_print_trainer_samples():
    seed = 42
    num_batches = 5
    # Set the seed for reproducibility
    reproducibility.seed_all(seed)

    # Create a dataset
    dataset = RandomClassificationDataset(size=100)

    # Create a dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        sampler=dist.get_sampler(dataset),
    )

    # Create a model
    model = SimpleModel()

    # Configuration for the trainer
    config = {
        'model': model,
        'train_dataloader': dataloader,
        'max_duration': f'{num_batches}ba',  # Train for specified number of batches
        'seed': seed,
    }

    # Create a trainer
    trainer = Trainer(**config)

    # List to store samples
    samples = []

    # Train and collect samples
    for batch in trainer.fit_iterator():
        sample = batch[0].clone()  # Assuming the first element of the batch is the input data
        samples.append(sample)
        print(f"Batch {len(samples)}:")
        print(sample)
        print()  # Add a blank line for readability

        if len(samples) == num_batches:
            break

    print(f"Collected {len(samples)} batches of samples.")
    assert False

