import torch.nn as nn
import logging
from arc_drone.auair_eval import apply_submodule_patch

def test_apply_submodule_patch():
    # Simulate older torch by deleting set_submodule if it exists
    original_set_submodule = None
    if hasattr(nn.Module, 'set_submodule'):
        original_set_submodule = nn.Module.set_submodule
        del nn.Module.set_submodule
    
    try:
        assert not hasattr(nn.Module, 'set_submodule')
        apply_submodule_patch()
        assert hasattr(nn.Module, 'set_submodule')
        
        # Test functionality
        class Sub(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)
        
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()
        
        p = Parent()
        new_layer = nn.Linear(5, 5)
        p.set_submodule("sub.layer", new_layer)
        assert p.sub.layer is new_layer
        
    finally:
        # Restore environment
        if original_set_submodule:
            nn.Module.set_submodule = original_set_submodule

