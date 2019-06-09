import implicit
from scipy import sparse
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import NormalPredictor
from .EvaluationData import EvaluationData
from .RecommenderMetrics import RecommenderMetrics


class ImplicitModels:

    def __init__(self, df, factors=50):
        self.df = df
        self.factors = factors
        reader = Reader(line_format='user item rating')
        data = Dataset.load_from_df(df=self.df, reader=reader)
        self.eval_data = EvaluationData(data)
        self.FullTrainSet = self.eval_data.GetFullTrainSet()
        self.n_item = self.FullTrainSet.n_items
        self.n_user = self.FullTrainSet.n_users
        self.model = implicit.als.AlternatingLeastSquares(self.factors)

    def get_sparse_matrix(self, TrainSet):
        rows = []
        cols = []
        data = []
        for iiid in TrainSet.ir:
            for uiid, rating in TrainSet.ir[iiid]:
                rows.append(iiid)
                cols.append(uiid)
                data.append(rating)
        sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_item, self.n_user))
        return sparse_matrix


    def topN_all(self, n=10, train=False):
        if train==False:
            TrainSet = self.FullTrainSet
        else:
            TrainSet = train
        sparse_matrix = self.get_sparse_matrix(TrainSet)
        self.model.fit(sparse_matrix)
        topN = defaultdict(list)
        for iuid in TrainSet.ur:
            predictions = self.model.recommend(iuid, sparse_matrix.T, N=n)
            for iiid, rating in predictions:
                ruid = self.FullTrainSet.to_raw_uid(iuid)
                riid = self.FullTrainSet.to_raw_iid(iiid)
                topN[ruid].append((riid, rating))

        return topN

    def topN_user(self, n=10, ruid=0):
        sparse_matrix = self.get_sparse_matrix(self.FullTrainSet)
        self.model.fit(sparse_matrix)
        iuid = self.FullTrainSet.to_inner_uid(ruid)
        topN = self.model.recommend(iuid, sparse_matrix.T, N=n)
        for i in topN:
            i[0] = self.FullTrainSet.to_raw_iid(i[0])

        return topN

    def user_seen(self, n=10, ruid=0):
        iuid = self.FullTrainSet.to_inner_uid(ruid)
        seen = [(self.FullTrainSet.to_raw_iid(i[0]), i[1]) for i in self.FullTrainSet.ur[iuid]]
        seen.sort(key=lambda x: x[1], reverse=True)
        return seen[:n]

    def eval(self, n=10, doTopN=True):
        metrics = {}
        if doTopN:
            LOOCVTrainSet = self.eval_data.GetLOOCVTrainSet()
            LOOCVTestSet = self.eval_data.GetLOOCVTestSet()
            TopN = self.topN_all(train=LOOCVTrainSet)
            metrics['HitRate'] = RecommenderMetrics.HitRate(TopN, LOOCVTestSet)

        return metrics