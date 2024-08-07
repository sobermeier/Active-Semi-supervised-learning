import logging
import pathlib
from typing import Dict, Any, Tuple, Sequence

import mlflow
import pandas as pd

from settings import DATA_ROOT

# originial mlflow logger code by Sandra Gilhuber

def flatten_dict(d):
	"""
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

	def expand(key, value):
		if isinstance(value, dict):
			return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
		else:
			return [(key, value)]

	items = [item for k, v in d.items() for item in expand(k, v)]

	return dict(items)


class MLFlowLogger:
	def __init__(
			self,
			root: str = DATA_ROOT + '/experiments',
			tracking_uri: str = 'http://usedom.dbs.ifi.lmu.de:5002/',
			experiment_name: str = 'SALOON_Test',
			artifact_location="/nfs/data8/jahnp/saloon/artifacts",
	):
		"""
        Constructor.

        Connects to a running MLFlow instance in which the runs of the experiments are stored.
        Also creates an output root directory. The directory will be created if it does not exist.

        :param root: str
            The path of the output root.
        :param tracking_uri: str
            The uri where the MLFlow instance is running.
        :param experiment_name: str
            The name of the experiment on the MLFlow instance server.
        """
		mlflow.set_tracking_uri(uri=tracking_uri)
		experiment = mlflow.get_experiment_by_name(name=experiment_name)
		if experiment is None:
			experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
		else:
			experiment_id = experiment.experiment_id
		self.experiment_id = experiment_id
		self.root = pathlib.Path(root)
		self.current_run = None
		self.log_file = None

	def init_experiment(
			self,
			name,
			hyper_parameters: Dict[str, Any],
			log_file=None,
			tracking_uri = mlflow.get_tracking_uri()
		) -> Tuple[str, str]:
		"""
        Initialise an experiment, i.e. a run in the specified experiment.
        Creates an entry describing the run's hyper-parameters, and associates an unique output directory.

        :param hyper_parameters: Dict
            The hyperparameters. Should be serialisable to JSON/BSON.
        :param log_file: file path, path to log_file
        :return: A tuple (id, path) where
            id:
                a unique ID of the experiment (equal to the ID in the MLFlow instance)
            path:
                the path to an existing directory for outputs of the experiment.
        """

		mlflow.set_tracking_uri(tracking_uri)
		self.current_run = mlflow.start_run(run_name=name, experiment_id=self.experiment_id)
		self.log_file = log_file
		# create output directory
		output_path = self.root / str(self.current_run.info.run_id)
		if output_path.is_dir():
			logging.error('Output path already exists! {p}'.format(p=output_path))
		output_path.mkdir(exist_ok=True, parents=True)

		print(type(output_path))
		print(output_path)
		#hyper_parameters["out_dir"] = output_path

		self.log_params(hyper_parameters=flatten_dict(vars(hyper_parameters)))
		# return id as experiment handle, and path of existing directory to store outputs to
		return self.current_run.info.run_id, str(output_path)

	@staticmethod
	def log_params(hyper_parameters: Dict[str, Any]):
		mlflow.log_params(params=flatten_dict(hyper_parameters))

	@staticmethod
	def finalise_experiment() -> None:
		"""
        Close the current run.
        :return: None.
        """
		mlflow.end_run()

	@staticmethod
	def log_results(result: Dict[str, Any], step=None):
		"""
        :param stat_file: Path to statistics file of current step
        :param log_file: Path to logging file
        :param result: Dict
            A flattened dictionary holding high-level results, e.g. a few numbers.
        :param step: int
        :return: None.
        """
		mlflow.log_metrics(metrics=flatten_dict(result), step=step)


	def log_log_file(self):
		if self.log_file is not None:
			mlflow.log_artifact(local_path=self.log_file)

	def get_entries(
			self,
			keys: Sequence[str],
			equals: Sequence[bool],
			values: Sequence[str]
	) -> pd.DataFrame:
		"""
        Get entries from Mlflow client in form of pandas dataframe.
        :param keys: keys to filter by.
        :param equals: whether key should be equal or not equal to value.
        :param values: values to filter by. The order must match <keys>
        :return: pd.DataFrame containing the result
        """
		search_string = ''
		for index, key in enumerate(keys):
			operand = '=' if equals[index] else '!='
			tmp_str = f'{key} {operand} "{values[index]}"'
			if index != 0:
				tmp_str = f' and {tmp_str}'
			search_string = search_string + tmp_str
		return mlflow.search_runs(experiment_ids=self.experiment_id, filter_string=search_string)

	def get_tracking_uri(self):
		return mlflow.get_tracking_uri()
