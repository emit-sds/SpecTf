__all__ = [
    'train_resnet',
    'train_xgb',
]

from cloud.cli import spectf

@spectf.group(
    help="Training commands for the ResNet and XGBoost comparison models."
)
def train_comparison():
    """
    Commands for training the ResNet and XGBoost comparison models.
    """
    pass

from cloud.comparison_models import train_resnet, train_xgb