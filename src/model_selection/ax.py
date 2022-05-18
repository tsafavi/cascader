import os
import shutil

from ax.service.ax_client import AxClient

from config import Config


class AxSearchJob(object):
    """A class for hyperparameter search jobs with Ax."""

    def __init__(self, config, main_fn):
        self.client = AxClient(verbose_logging=False, random_seed=54321)
        self.full_config = config  # the base config for the spawned jobs
        self.main_fn = main_fn

        self.parameters = config.get("search.parameters")
        self.num_trials = config.get("search.num_trials")
        self.monitor_metric = config.get("monitor.metric")
        self.monitor_mode = config.get("monitor.mode")

        self.do_checkpoint = config.get("do-checkpoint")

    @staticmethod
    def create(config, main_fn):
        return AxSearchJob(config, main_fn)

    def run(self):
        self.client.create_experiment(
            parameters=self.parameters,
            objective_name=self.monitor_metric,
            minimize=(self.monitor_mode == "min"),
        )

        for i in range(self.num_trials):
            parameters, trial_index = self.client.get_next_trial()
            trial_str = f"trial_{str(trial_index).zfill(4)}"
            print(f"Trial {trial_str} parameters: {parameters}")

            # Create a new job and config
            trial_folder = os.path.join(self.full_config.folder, trial_str)
            if self.do_checkpoint:
                os.makedirs(trial_folder, exist_ok=True)

            trial_config = Config.create_from_config(
                self.full_config, folder=trial_folder
            )

            # Set new parameters in the original config
            for key, value in parameters.items():
                trial_config.set(key, value)

            if self.do_checkpoint:
                trial_config.save_as_yaml()

            raw_data = self.main_fn(trial_config)

            self.client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        trials_df = self.client.get_trials_data_frame()
        trials_df["trial"] = [
            f"trial_{str(trial_index).zfill(4)}"
            for trial_index in range(self.num_trials)
        ]
        trials_df.to_csv(os.path.join(self.full_config.folder, "trials.csv"))

        best_trial_idx = trials_df[self.monitor_metric].argmax()
        best_trial_idx_str = str(best_trial_idx).zfill(4)
        print("Best trial:", best_trial_idx_str)
        # print("Best parameters:", self.client.get_best_parameters()[0])

        if self.do_checkpoint:
            trial_dirs = os.listdir(self.full_config.folder)
            for trial_dir in trial_dirs:
                trial_folder = os.path.join(self.full_config.folder, trial_dir)
                if os.path.isdir(trial_folder) and not trial_folder.endswith(best_trial_idx_str):
                    shutil.rmtree(trial_folder)
                    print(f"Deleted trial folder {trial_folder}")
                elif os.path.isdir(trial_folder) and trial_folder.endswith(best_trial_idx_str):
                    checkpoint_dir = os.path.join(trial_folder, "checkpoints")
                    src_path = os.path.join(checkpoint_dir, "checkpoint_best.ckpt")
                    dst_path = os.path.join(trial_folder, "checkpoint_best.ckpt")
                    shutil.move(src_path, dst_path)
                    shutil.rmtree(checkpoint_dir)
                    new_trial_folder = os.path.join(self.full_config.folder, "checkpoints")
                    os.rename(trial_folder, new_trial_folder)
                    print(f"Renamed {trial_folder} -> {new_trial_folder}")

        cols = [
            "mrr",
            "mr",
            "hits@1",
            "hits@3",
            "hits@10",
            "test_mrr",
            "test_mr",
            "test_hits@1",
            "test_hits@3",
            "test_hits@10",
        ]

        if set(cols).issubset(trials_df.columns):
            values = trials_df.iloc[best_trial_idx][
                [
                    "mrr",
                    "mr",
                    "hits@1",
                    "hits@3",
                    "hits@10",
                    "test_mrr",
                    "test_mr",
                    "test_hits@1",
                    "test_hits@3",
                    "test_hits@10",
                ]
            ].values

            print(",".join([f"{val:.5f}" for val in values]))
