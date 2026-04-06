import subprocess
import os

def test_benchmark_args():
    with open("tests/test_wrapper.py", "w") as f:
        f.write('''
import sys
import unittest.mock
import scripts.benchmark_auair as benchmark_auair

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

benchmark_auair.evaluate_gemma = unittest.mock.Mock(return_value=m)
benchmark_auair.evaluate_trm = unittest.mock.Mock(return_value=m)
benchmark_auair.adapt_and_evaluate_gemma = unittest.mock.Mock(return_value=m)

sys.argv = ['benchmark_auair.py', '--sequences', 'data/auair_sequences.jsonl', '--images-path', '/workspace/gemma-4/images', '--model-id', 'google/gemma-4-e4b-it', '--n-eval', '300']
try:
    benchmark_auair.main()
except SystemExit as e:
    sys.exit(e.code)
sys.exit(0)
''')
    res = subprocess.run(["python", "tests/test_wrapper.py"], capture_output=True, text=True)
    print("STDERR", res.stderr)
    print("STDOUT", res.stdout)
    assert res.returncode == 0

