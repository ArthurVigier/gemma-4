import sys
import unittest.mock
import pytest
import argparse

import scripts.benchmark_auair as benchmark_auair
import scripts.finetune_gemma_auair as finetune_gemma_auair
import scripts.train_trm_student as train_trm_student
import scripts.distill_trm_student as distill_trm_student

def test_benchmark_auair_args():
    m = unittest.mock.Mock()
    m.model_name = "test"
    m.action_acc = 1.0
    m.parse_rate = 1.0
    m.halt_acc = 1.0
    m.ms_per_sample = 1.0
    m.n_samples = 1
    m.n_heuristic = 1
    m.extra = {}
    m.chunk_acc = [1.0, 1.0, 1.0, 1.0]

    with unittest.mock.patch('scripts.benchmark_auair.evaluate_gemma', return_value=m), \
         unittest.mock.patch('scripts.benchmark_auair.evaluate_trm', return_value=m), \
         unittest.mock.patch('scripts.benchmark_auair.adapt_and_evaluate_gemma', return_value=m):
        
        sys.argv = [
            'benchmark_auair.py', 
            '--sequences', 'data/auair_sequences.jsonl', 
            '--images-path', '/workspace/gemma-4/images', 
            '--model-id', 'google/gemma-4-e4b-it', 
            '--n-eval', '300'
        ]
        
        try:
            benchmark_auair.main()
        except SystemExit as e:
            assert e.code == 0

def test_finetune_gemma_auair_args():
    with unittest.mock.patch('scripts.finetune_gemma_auair.finetune_auair_teacher') as mock_finetune:
        mock_finetune.return_value = {"best_eval_loss": 0.1, "output_dir": "test_dir"}
        sys.argv = [
            'finetune_gemma_auair.py',
            '--auair-path', 'data/auair_sequences.jsonl',
            '--auair-images-path', '/workspace/gemma-4/images',
            '--epochs', '3'
        ]
        try:
            finetune_gemma_auair.main()
        except SystemExit as e:
            assert e.code == 0
        mock_finetune.assert_called_once()

def test_train_trm_student_args():
    m = unittest.mock.Mock()
    m.best_eval_action_accuracy = 1.0
    m.best_eval_halt_step_mae = 1.0
    m.output_dir = "test_dir"
    m.onnx_output_path = None
    m.trtexec_command = None
    m.foundation_model_id = "test"
    m.device = "cuda"
    m.parameter_count_millions = 1.0

    with unittest.mock.patch('scripts.train_trm_student.train_student', return_value=m):
        sys.argv = [
            'train_trm_student.py',
            '--auair-path', 'data/auair_sequences.jsonl',
            '--auair-images-path', '/workspace/gemma-4/images'
        ]
        try:
            train_trm_student.main()
        except SystemExit as e:
            assert e.code == 0

def test_distill_trm_student_args():
    m = unittest.mock.Mock()
    m.best_eval_action_accuracy = 1.0
    m.best_eval_halt_step_mae = 1.0
    m.output_dir = "test_dir"
    m.onnx_output_path = None
    m.trtexec_command = None
    m.foundation_model_id = "test"
    m.device = "cuda"
    m.parameter_count_millions = 1.0

    with unittest.mock.patch('scripts.distill_trm_student.distill_student', return_value=m):
        sys.argv = [
            'distill_trm_student.py',
            '--auair-path', 'data/auair_sequences.jsonl',
            '--auair-images-path', '/workspace/gemma-4/images',
            '--teacher-lora-path', 'artifacts/teacher_lora/gemma_e4b_auair'
        ]
        try:
            distill_trm_student.main()
        except SystemExit as e:
            assert e.code == 0
