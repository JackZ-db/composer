import torch
from torch.utils.data import DataLoader
from composer import Trainer
from composer.models import ComposerModel
from tests.common import SimpleModel, RandomClassificationDataset
from composer.utils import dist, reproducibility
import pytest 
from composer.loggers import InMemoryLogger, Logger, RemoteUploaderDownloader


from composer import Callback, Evaluator, Trainer
from composer.algorithms import CutOut, LabelSmoothing
from composer.core import Event, Precision, State, Time, TimeUnit
from composer.devices import Device

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
        batch_size=8,
        sampler=dist.get_sampler(dataset),
    )

    # Create a model
    model = SimpleModel()
    class SampleCollector(Callback):
        def after_train_batch(self, state: State, logger: Logger):
            samples.append(state.batch[0].clone())

    sampleCollector = SampleCollector()

    # Create a trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        max_duration=10,
        device_train_microbatch_size=2,
        callbacks=[sampleCollector],
        seed=42
        )

    # List to store samples
    samples = []

    
    
    trainer.fit()
    
    for i, sample in enumerate(samples, 1):
        print(f"Batch {i}:")
        print(sample)
        print()
    
    print(f"Collected {len(samples)} batches of samples.")

    print(f"Collected {len(samples)} batches of samples.")
    assert False

