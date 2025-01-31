import xgboost
import yaml
from typing import List

# Example:
# model = make_xgb_model(arch_yml='arch.yml', weight_file='xgboost.json', arch_subkeys=["architecture"])
def make_xgb_model(arch_yml:str, weight_file:str, arch_subkeys:List[str]=[]):
	with open(arch_yml, 'r') as f:
		arch = yaml.safe_load(f)
		if arch_subkeys:
			for key in arch_subkeys:
				arch = arch[key]
                
	model = xgboost.XGBClassifier(
		**arch
	)
	model.load_model(weight_file)
	return model