from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from .EvaluationData import EvaluationData
from .RecommenderMetrics import RecommenderMetrics


class ExplicitModels:

    def __init__(self, df, algo='KNN', user_based=False):
        self.df = df
        self.algo = algo
        self.user_based = user_based

        reader = Reader(line_format='user item rating')
        data = Dataset.load_from_df(df=self.df, reader=reader)
        self.eval_data = EvaluationData(data)

        if self.algo == 'KNN':
            sim_options = {'name': 'cosine', 'user_based': self.user_based}
            self.model = KNNBasic(sim_options=sim_options)
        elif self.algo == 'SVD':
            self.model = SVD()
        elif self.algo == 'SVD++':
            self.model = SVDpp()
        elif self.algo == 'Random':
            self.model = NormalPredictor()

    def predict_all(self, n=10):
        FullTrainSet = self.eval_data.GetFullTrainSet()
        FullAntiTestSet = self.eval_data.GetFullAntiTestSet()
        self.model.fit(FullTrainSet)
        allPredictions = self.model.test(FullAntiTestSet)
        topN = RecommenderMetrics.GetTopN(allPredictions, n=n)

        return topN

    def predict_user(self, n=10, ruid=85):
        FullTrainSet = self.eval_data.GetFullTrainSet()
        UserAntiTestSet = self.eval_data.GetAntiTestSetForUser(ruid)
        self.model.fit(FullTrainSet)
        predictions = self.model.test(UserAntiTestSet)

        recommendations = []
        for userID, itemID, actualRating, estimatedRating, _ in predictions:
            recommendations.append((itemID, estimatedRating))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]

    def user_seen(self, n=10, ruid=0):
        FullTrainSet = self.eval_data.GetFullTrainSet()
        iuid = FullTrainSet.to_inner_uid(ruid)
        seen = [(FullTrainSet.to_raw_iid(i[0]), i[1]) for i in FullTrainSet.ur[iuid]]
        seen.sort(key=lambda x: x[1], reverse=True)
        return seen[:n]

    def eval(self, n=10, doTopN=True):
        metrics = {}
        TrainSet = self.eval_data.GetTrainSet()
        TestSet = self.eval_data.GetTestSet()
        self.model.fit(TrainSet)
        predictions = self.model.test(TestSet)
        metrics['RMSE'] = RecommenderMetrics.RMSE(predictions)
        metrics['MAE'] = RecommenderMetrics.MAE(predictions)

        if doTopN:
            LOOCVTrainSet = self.eval_data.GetLOOCVTrainSet()
            LOOCVTestSet = self.eval_data.GetLOOCVTestSet()
            LOOCVAntiTestSet = self.eval_data.GetLOOCVAntiTestSet()
            self.model.fit(LOOCVTrainSet)
            predictions = self.model.test(LOOCVAntiTestSet)
            TopN = RecommenderMetrics.GetTopN(predictions, n=n)
            metrics['HitRate'] = RecommenderMetrics.HitRate(TopN, LOOCVTestSet)

        return metrics