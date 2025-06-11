
def enable_nonsac(enable_remove_add=True):
    import torch._functorch.config
    import torch._functorch.partitioners as partitioners
    torch._functorch.config.activation_memory_budget = 0.99

    if enable_remove_add:
        # Remove 'add' from the default operation list to eliminate redundant re-computation 
        # due to wrong partitioning of nodes.
        def remove_add(fn):
            def wrapped_fn():
                optypes = fn()
                optypes.recomputable_ops.remove(torch.ops.aten.add)
                return optypes
            return wrapped_fn

        partitioners.get_default_op_list = remove_add(partitioners.get_default_op_list)