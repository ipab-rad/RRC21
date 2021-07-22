from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.distances import CosineSimilarity

from learner import Embedder
from utils import  GraspMetricDataset


class GraspEvaluator(object):

    def __init__(self,
                k,
                model_path,
                data_path):
        """Grasp Evaluator based on the learned metric
        :param k: number of neigbours to return 
        :param model_path: path to embedder model state_dict
        :param data_path: path to the dictionary of position - grasp list
        """

        self.k = k
        self.dataset = GraspMetricDataset(data_path)
        self.model = InferenceModel(trunk=Embedder.load_state_dict(model_path),
                                match_finder=MatchFinder(distance=CosineSimilarity(), threshold=0.7))

        self.model.train_indexer(self.dataset)

    
    def __call__(self, position):

        idxs, distances = self.model.get_nearest_neighbors(position, self.k)

        grasps = [self.dataset[i][0] for i in idxs[0]]

        return grasps, distances