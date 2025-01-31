__all__ = [
    'eval_l2a',
    'eval_spectf',
]

from spectf_cloud.cli import spectf_cloud

@spectf_cloud.group(
    help="Evaluation commands for SpecTf and L2A Baseline."
)
def cloud_eval():
    """
    A group of evaluation-related subcommands, e.g., 'cloud_eval eval_l2a'.
    """
    pass

from spectf_cloud.evaluation import eval_l2a, eval_spectf