import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
from torch import Tensor, nn
import subprocess
import wandb

from click.testing import CliRunner

from cloud.train_spectf_cloud import train
from cloud.model import SimpleSeqClassifier
from make_dummy_data import NUM_DATAPOINTS

WANDB_PATH = "cloud.train_spectf_cloud.wandb"
DUMMY_DATA = "data/mock_dataset.hdf5"

class TestSpecTfTrain(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.base = os.path.dirname(__file__)

        if not os.path.exists(os.path.join(self.base, DUMMY_DATA)):
            print("Creating mock dataset...")
            subprocess.run(["python3", os.path.join(self.base, "make_dummy_data.py")])


    @patch(WANDB_PATH)
    def test_train_command(self, mock_wandb):
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
        with patch.object(SimpleSeqClassifier, "forward", side_effect=mock_forward_impl) as mock_forward:
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
                
            
if __name__ == "__main__":
    unittest.main()