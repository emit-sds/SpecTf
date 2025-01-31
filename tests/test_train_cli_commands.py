import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
from torch import Tensor, nn
import numpy as np
import subprocess
import wandb

from click.testing import CliRunner

from spectf_cloud.train import train
from spectf_cloud.comparison_models.train_resnet import resnet
from spectf_cloud.comparison_models.train_xgb import xgboost
from spectf.model import SpecTfEncoder
from spectf_cloud.comparison_models.ResNet import ResNet
from spectf_cloud.comparison_models.train_xgb import xgb
from make_dummy_data import NUM_DATAPOINTS

SPECTF_WANDB_PATH = "spectf_cloud.train.wandb"
RESNET_WANDB_PATH = "spectf_cloud.comparison_models.train_resnet.wandb"
XGBOOST_WANDB_PATH = "spectf_cloud.comparison_models.train_xgb.wandb"
RESNET_OPTIMIZER = "spectf_cloud.comparison_models.train_resnet.AdamWScheduleFree"
DUMMY_DATA = "data/mock_dataset.hdf5"

class TestTrainCommands(unittest.TestCase):
    """
    :TestTrainCommands:

    The purpose of this test suite is to test to see if the CLI functions generally work so that they're executable.
    NOT to test the following:
    • CLI parameter functions
    • Model results
    """

    def setUp(self):
        self.runner = CliRunner()
        self.base = os.path.dirname(__file__)

        if not os.path.exists(os.path.join(self.base, DUMMY_DATA)):
            print("Creating mock dataset...")
            subprocess.run(["python3", os.path.join(self.base, "make_dummy_data.py")])


    @patch(SPECTF_WANDB_PATH)
    def test_spectf_train_command(self, mock_wandb):
        """
        Test the 'spectf train' CLI with dummy dataset and mock model.
        """
        epochs = 2

        # 1. Setup mocks
        # Mock wandb
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.Settings = wandb.Settings

        # Mock model forward function call
        def mock_forward_impl(x:Tensor) -> Tensor:
            """
            A custom forward that does a simple linear projection from x.size(-1)
            to 2 classes, for example. Or just returns random logits.
            """
            # We can define a small linear layer on the fly:
            x = x.squeeze(-1)
            linear = nn.Linear(x.size(-1), 2, bias=True).to(x.device)
            out = linear(x.float())
            return out

        # 2. Invoke the CLI
        with patch.object(SpecTfEncoder, "forward", side_effect=mock_forward_impl) as mock_forward:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self.runner.invoke(
                    train,
                    [
                        os.path.join(self.base, DUMMY_DATA),
                        "--train-csv", os.path.join(self.base, "data/mock_train.csv"),
                        "--test-csv", os.path.join(self.base, "data/mock_test.csv"),
                        "--epochs", str(epochs),
                        "--outdir", tmpdir,
                        "--batch", str(NUM_DATAPOINTS),
                    ],
                )

                if result.exception:
                    raise result.exception

                # Should exit successfully
                self.assertEqual(result.exit_code, 0, msg=f"CLI failed: {result.output}") 

                # We expect # epochs * 3 forward calls (1 for training and 2 for validation on train/test)
                self.assertEqual(
                    mock_forward.call_count, 
                    epochs*3, 
                    f"Expected forward() to be called {epochs*3} times, got {mock_forward.call_count}"
                )       

                # Check the files written to the output directory (shows run finished and model saved)
                files_written = os.listdir(tmpdir)
                self.assertEqual(len(files_written), 1, msg="Expected 1 file written. Got: "+str(files_written))


    @patch(RESNET_WANDB_PATH)
    @patch(RESNET_OPTIMIZER)
    def test_resnet_train_command(self, mock_wandb, mock_optimizer):
        """
        Test the 'spectf train-comparison resnet' CLI with dummy dataset and mock model.
        """
        epochs = 2

        # 1. Setup mocks
        # Mock wandb
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.Settings = wandb.Settings
        mock_optimizer = MagicMock()

        # Mock model forward function call
        def mock_forward_impl(x:Tensor) -> Tensor:
            """
            A custom forward that does a simple linear projection from x.size(-1)
            to 2 classes, for example. Or just returns random logits.
            """
            # We can define a small linear layer on the fly:
            x = x.squeeze(-1)
            linear = nn.Linear(x.size(-1), 2, bias=True).to(x.device)
            linear.train()
            out = linear(x.float())
            return out

        # 2. Invoke the CLI
        with patch.object(ResNet, "forward", side_effect=mock_forward_impl) as mock_forward:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self.runner.invoke(
                    resnet,
                    [
                        os.path.join(self.base, DUMMY_DATA),
                        "--train-csv", os.path.join(self.base, "data/mock_train.csv"),
                        "--test-csv", os.path.join(self.base, "data/mock_test.csv"),
                        "--arch-yaml", os.path.join(os.path.dirname(self.base), "spectf_cloud/comparison_models/ResNet/resnet_arch.yml"),
                        "--epochs", str(epochs),
                        "--outdir", tmpdir,
                        "--batch", str(NUM_DATAPOINTS),
                    ],
                )
                # Should exit successfully
                self.assertEqual(result.exit_code, 0, msg=f"CLI failed: {result.output}") 

                # We expect # epochs*2+1 forward calls 
                self.assertEqual(
                    mock_forward.call_count, 
                    epochs*2+1, 
                    f"Expected forward() to be called {epochs*2+1} times, got {mock_forward.call_count}"
                )       

                # Check the files written to the output directory (shows run finished and model saved)
                # Expect 3 - the log, the model weights, and the savd eval scores
                files_written = os.listdir(tmpdir)
                self.assertEqual(len(files_written), 3, msg="Expected 3 file written. Got: "+str(files_written))


    @patch(XGBOOST_WANDB_PATH)
    @patch.object(xgb.XGBClassifier, "save_model")
    @patch.object(xgb.XGBClassifier, "predict_proba")
    @patch.object(xgb.XGBClassifier, "fit")
    def test_xgboost_train_command(
        self,
        mock_fit, 
        mock_predict_proba,
        mock_save_model, 
        mock_wandb
    ):
        """
        Test the 'spectf train-comparison xgboost' CLI with dummy dataset and mock model.
        """

        # 1. Setup mocks
        # Mock wandb
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.Settings = wandb.Settings

        # 2. Define side effects (fake implementations)
        mock_fit.side_effect = lambda X, y: None
        mock_predict_proba.side_effect = (lambda X: np.ones((X.shape[0], 2), dtype=np.float32))
        mock_save_model.side_effect = lambda filename: None

        # 2. Invoke the CLI
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                xgboost,
                [
                    os.path.join(self.base, DUMMY_DATA),
                    "--train-csv", os.path.join(self.base, "data/mock_train.csv"),
                    "--test-csv", os.path.join(self.base, "data/mock_test.csv"),
                    "--arch-yaml", os.path.join(os.path.dirname(self.base), "spectf_cloud/comparison_models/XGBoost/xgboost_arch.yml"),
                    "--outdir", tmpdir,
                ],
            )
            # Should exit successfully
            self.assertEqual(result.exit_code, 0, msg=f"CLI failed: {result.output}") 

            # We expect 1 fit call
            self.assertEqual(
                mock_fit.call_count, 
                1, 
                f"Expected fit() to be called 1 time, got {mock_fit.call_count}"
            )

            # Check the files written to the output directory (shows run finished and model saved)
            # Expect 2 files - the log and the F-Beta eval scores
            files_written = os.listdir(tmpdir)
            self.assertEqual(len(files_written), 2, msg="Expected 2 file written. Got: "+str(files_written))
                
            
if __name__ == "__main__": 
    unittest.main()
