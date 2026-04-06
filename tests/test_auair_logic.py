import pytest
from arc_drone.auair_eval import _parse_chunk

def test_parse_chunk_flexible():
    # Test different formatting that Gemma-4 might produce
    text1 = "Action_0: 4  Halt_0: 2\nAction_1: 5  Halt_1: 3"
    a, h = _parse_chunk(text1, 2)
    assert a == [4, 5]
    assert h == [2, 3]

    text2 = "Action 0: 1, Halt 0: 6. Action 1: 0, Halt 1: 1."
    a, h = _parse_chunk(text2, 2)
    assert a == [1, 0]
    assert h == [6, 1]

    text3 = "ACTION_0: 7\nHALT_0: 1"
    a, h = _parse_chunk(text3, 1)
    assert a == [7]
    assert h == [1]

def test_gt_labels_logic():
    # Simulate the dictionary structure from sequences
    seq = {
        "action_index": 5,
        "halt_step": 3,
        "action_indices": [1, 2],
        "halt_steps": [4, 4]
    }
    C = 4
    # Replicate the logic in the loop
    gt_actions_raw = seq.get("action_indices", [seq.get("action_index", 0)] * C)
    gt_actions = (list(gt_actions_raw) * C)[:C]
    assert gt_actions == [1, 2, 1, 2]
    
    seq_no_list = {"action_index": 5, "halt_step": 3}
    gt_actions_raw2 = seq_no_list.get("action_indices", [seq_no_list.get("action_index", 0)] * C)
    gt_actions2 = (list(gt_actions_raw2) * C)[:C]
    assert gt_actions2 == [5, 5, 5, 5]

