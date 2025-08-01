# -*- coding: utf-8 -*-
import csv
import json
import os
import tempfile
import numpy as np
import pytest

from scilpy.stats.utils import data_for_stat, get_group_data_sample


@pytest.fixture
def dummy_stats_files():
    json_data = {
        "sub-01": {"bundle1": {"metric1": {"value1": 0.5}}},
        "sub-02": {"bundle1": {"metric1": {"value1": 0.6}}}
    }
    tsv_data = [
        {"participant_id": "sub-01", "group": "A"},
        {"participant_id": "sub-02", "group": "B"}
    ]

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, newline='') as f_json, \
         tempfile.NamedTemporaryFile(
             mode='w', suffix='.tsv', delete=False, newline='') as f_tsv:
        json.dump(json_data, f_json)
        f_json.flush()

        writer = csv.DictWriter(f_tsv, fieldnames=["participant_id", "group"],
                                delimiter=' ')
        writer.writeheader()
        writer.writerows(tsv_data)
        f_tsv.flush()

        yield f_json.name, f_tsv.name

    os.remove(f_json.name)
    os.remove(f_tsv.name)


def test_data_for_stat_init(dummy_stats_files):
    json_file, tsv_file = dummy_stats_files
    stats_data = data_for_stat(json_file, tsv_file)

    assert set(stats_data.get_participants_list()) == {"sub-01", "sub-02"}
    assert stats_data.get_bundles_list() == ["bundle1"]
    assert stats_data.get_metrics_list() == ["metric1"]
    assert stats_data.get_values_list() == ["value1"]
    assert stats_data.get_participant_attributes_list() == ["group"]


def test_data_for_stat_init_mismatch(dummy_stats_files):
    json_file, tsv_file = dummy_stats_files

    # Create a mismatched tsv
    with open(tsv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "group"],
                                delimiter=' ')
        writer.writeheader()
        writer.writerow({"participant_id": "sub-03", "group": "A"})

    with pytest.raises(BaseException):
        data_for_stat(json_file, tsv_file)


def test_get_groups_dictionary(dummy_stats_files):
    json_file, tsv_file = dummy_stats_files
    stats_data = data_for_stat(json_file, tsv_file)

    groups = stats_data.get_groups_dictionnary("group")
    assert set(groups.keys()) == {"group_A", "group_B"}
    assert list(groups["group_A"].keys()) == ["sub-01"]
    assert list(groups["group_B"].keys()) == ["sub-02"]

    with pytest.raises(BaseException):
        stats_data.get_groups_dictionnary("non_existent_group")


def test_get_group_data_sample(dummy_stats_files):
    json_file, tsv_file = dummy_stats_files
    stats_data = data_for_stat(json_file, tsv_file)
    groups = stats_data.get_groups_dictionnary("group")

    sample_A = get_group_data_sample(groups, "group_A", "bundle1",
                                     "metric1", "value1")
    assert np.array_equal(sample_A, [0.5])

    sample_B = get_group_data_sample(groups, "group_B", "bundle1",
                                     "metric1", "value1")
    assert np.array_equal(sample_B, [0.6])


def test_write_current_dictionnary():
    # TODO: Implement this test
    pass


def test_write_csv_from_json():
    # TODO: Implement this test
    pass


def test_visualise_distribution():
    # TODO: Implement this test
    pass
