__all__ = [
    'train_resnet',
    'train_xgb',
]

from spectf_cloud.cli import spectf_cloud, MAIN_CALL_ERR_MSG

@spectf_cloud.group(
    help="Training commands for the ResNet and XGBoost comparison models."
)
def train_comparison():
    """
    Commands for training the ResNet and XGBoost comparison models.
    """
    pass

from spectf_cloud.comparison_models import train_resnet, train_xgb