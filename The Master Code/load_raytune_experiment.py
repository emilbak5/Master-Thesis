from ray import tune, air
from ray.tune import ResultGrid, ExperimentAnalysis






EXPERIMENT_PATH = 'C:/Users/emilb/ray_results/tune_sticker/experiment_state-2023-04-17_15-27-54.json'


print(f"Loading results from {EXPERIMENT_PATH}...")


restored_tuner = ExperimentAnalysis(EXPERIMENT_PATH)
result_grid = restored_tuner.get_results()