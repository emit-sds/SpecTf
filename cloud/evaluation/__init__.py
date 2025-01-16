__all__ = [
    'eval_l2a',
    'eval_spectf',
]

from cloud.cli import spectf

@spectf.group(
    help="Evaluation commands for SpecTf and L2A Baseline."
)
def eval():
    """
    A group of evaluation-related subcommands, e.g., 'eval eval_l2a'.
    """
    pass

from cloud.evaluation import eval_l2a, eval_spectf