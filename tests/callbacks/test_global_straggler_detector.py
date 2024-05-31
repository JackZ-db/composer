
import collections.abc
import datetime

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from composer.callbacks import GlobalStragglerDetector
from composer.core import Time
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel
from composer.utils import dist, get_device

def to_float(string, unit):
    index = string.find(unit)
    if index != -1:
        substring = string[:index]
        substring = substring.strip()
        float_value = float(substring)
        return float_value
    return 0.0


def _assert_no_negative_values(logged_values, unit):
    for timestamp, v in logged_values:
        del timestamp  # unused
        v = to_float(str(v), unit)
        if isinstance(v, Time):
            assert int(v) >= 0
        elif isinstance(v, datetime.timedelta):
            assert v.total_seconds() >= 0
        else:
            assert v >= 0

def _assert_leq_hundred(logged_values, unit):
    for timestamp, v in logged_values:
        del timestamp  # unused
        v = to_float(str(v), unit)
        assert v <= 100.00


@pytest.mark.parametrize('flops_per_batch', [True])
@pytest.mark.gpu
@pytest.mark.world_size(2)
def test_global_straggler_detector(flops_per_batch: bool):
    dist.initialize_dist(get_device(None))

    # Construct the callbacks
    global_straggler_detector = GlobalStragglerDetector()
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    model = SimpleModel()
    if flops_per_batch:
        model.flops_per_batch = lambda batch: len(batch) * 100.0  # pyright: ignore[reportGeneralTypeIssues]
    
    # Construct the trainer and train
    dataset = RandomClassificationDataset()
    trainer = Trainer(
        model=model,
        callbacks=global_straggler_detector,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(dataset=dataset, sampler=DistributedSampler(dataset=dataset)),
        max_duration='1ep',
    )
    trainer.fit()

    print("Length of logger.data: " + str(len(in_memory_logger.data)))
    for key in in_memory_logger.data.keys():
        print(key)

    _assert_no_negative_values(in_memory_logger.data['MinRoundTripTime/Rank'], "ms")
    _assert_no_negative_values(in_memory_logger.data['MaxRoundTripTime/Rank'], "ms")
    _assert_no_negative_values(in_memory_logger.data['MinPower/Rank'], "W")
    _assert_no_negative_values(in_memory_logger.data['MaxPower/Rank'], "W")
    _assert_no_negative_values(in_memory_logger.data['MinTemp/Rank'], "C")
    _assert_no_negative_values(in_memory_logger.data['MaxTemp/Rank'], "C")
    _assert_no_negative_values(in_memory_logger.data['MinUtilization/Rank'], "%")
    _assert_no_negative_values(in_memory_logger.data['MaxUtilization/Rank'], "%")
    _assert_leq_hundred(in_memory_logger.data['MinUtilization/Rank'], "%")
    _assert_leq_hundred(in_memory_logger.data['MaxUtilization/Rank'], "%")
    _assert_no_negative_values(in_memory_logger.data['MinClock/Rank'], "MHz")
    _assert_no_negative_values(in_memory_logger.data['MaxClock/Rank'], "MHz")
    _assert_no_negative_values(in_memory_logger.data['MinBatchLoadLatency/Rank'], "us")
    _assert_no_negative_values(in_memory_logger.data['MaxBatchLoadLatency/Rank'], "us")
    _assert_no_negative_values(in_memory_logger.data['MinThroughput/Rank'], "TF")
    _assert_no_negative_values(in_memory_logger.data['MaxThroughput/Rank'], "TF")

    

"""
def test_global_straggler_detector_tokens():
    model = SimpleTransformerClassifier()
    dataloader = dummy_text_classification_dataloader()
    dataloader.dataset.max_seq_len = dataloader.dataset.sequence_length  # type: ignore
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    global_straggler_detector = GlobalStragglerDetector()
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        callbacks=global_straggler_detector,
        loggers=in_memory_logger,
        max_duration='1ep',
    )
    trainer.fit()


    _assert_no_negative_values(in_memory_logger.data['MinRoundTripTime/Rank'], "ms")
    _assert_no_negative_values(in_memory_logger.data['MaxRoundTripTime/Rank'], "ms")
    _assert_no_negative_values(in_memory_logger.data['MinPower/Rank'], "W")
    _assert_no_negative_values(in_memory_logger.data['MaxPower/Rank'], "W")
    _assert_no_negative_values(in_memory_logger.data['MinTemp/Rank'], "C")
    _assert_no_negative_values(in_memory_logger.data['MaxTemp/Rank'], "C")
    _assert_no_negative_values(in_memory_logger.data['MinUtilization/Rank'], "%")
    _assert_no_negative_values(in_memory_logger.data['MaxUtilization/Rank'], "%")
    _assert_leq_hundred(in_memory_logger.data['MinUtilization/Rank'], "%")
    _assert_leq_hundred(in_memory_logger.data['MaxUtilization/Rank'], "%")
    _assert_no_negative_values(in_memory_logger.data['MinClock/Rank'], "MHz")
    _assert_no_negative_values(in_memory_logger.data['MaxClock/Rank'], "MHz")
    _assert_no_negative_values(in_memory_logger.data['MinBatchLoadLatency/Rank'], "us")
    _assert_no_negative_values(in_memory_logger.data['MaxBatchLoadLatency/Rank'], "us")
    _assert_no_negative_values(in_memory_logger.data['MinThroughput/Rank'], "TF")
    _assert_no_negative_values(in_memory_logger.data['MaxThroughput/Rank'], "TF")

    assert isinstance(trainer.state.dataloader, collections.abc.Sized)
    assert trainer.state.dataloader_label is not None
    assert trainer.state.dataloader_len is not None
    expected_step_calls = (trainer.state.dataloader_len - len(global_straggler_detector.history_samples) +
                           1) * int(trainer.state.timestamp.epoch)
    assert len(in_memory_logger.data['MinRoundTripTime/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxRoundTripTime/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MinPower/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxPower/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MinTemp/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxTemp/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MinUtilization/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxUtilization/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MinClock/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxClock/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MinBatchLoadLatency/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxBatchLoadLatency/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MinThroughput/Rank']) == expected_step_calls
    assert len(in_memory_logger.data['MaxThroughput/Rank']) == expected_step_calls
"""