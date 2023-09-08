import matplotlib.pyplot as plt
import numpy as np
import libNearestNeighbor
import logging
from metric_learn import MLKR
from scipy.linalg import sqrtm


class SAM_MLKR:

    def __init__(self, k=5, max_STM_size=250, max_LTM_size=500, use_MLKR=True, metric_step=100, metric_smoothing=False, metric_reducing=False, relevance_threshold=0.5, reducing_dimension = None):

        self.k = k
        self.max_LTM_size = max_LTM_size
        self.max_STM_size = max_STM_size

        self.STM_X = []
        self.STM_y = []
        self.LTM_X = []
        self.LTM_y = []

        self.y = []

        self.LTMPredHistory = []
        self.STMPredHistory = []
        self.CMPredHistory = []
        self.ModelPredHistory = []
        self.ModelPredictions = []
        self.LTMPredictions = []
        self.STMPredictions = []
        self.CMPredictions = []
        self.ModelError = None
        self.LTMError = None
        self.STMError = None
        self.CMError = None

        self.regressorChoice = []

        self.trainStepCount = 0
        self.STM_adaptions = 0
        self.STM_adaptions_idx = []
        self.STM_sizes = []
        self.LTM_sizes = []

        self.use_MLKR = use_MLKR
        self.metric_init = False
        self.metric_step = metric_step
        #self.metric_learner = MLKR(random_state=42)
        self.metric_learner = MLKR()
        self.metric_history = []
        self.metric_history_smoothed = []
        self.metric_history_reduced = []
        self.metric_smoothing = metric_smoothing
        self.metric_smoothing_window = 10
        self.metric_reducing = metric_reducing
        self.reducing_indices = []
        self.relevance_threshold = relevance_threshold
        self.reducing_dimension = reducing_dimension
        self.relevancies = []
        self.relevancies_normalized = []
        self.relevancies_reduced = []
        self.relevancy_times = []

        self.flag = False
        self.flag_occurence = 0


    def _partial_fit(self, x, y):

        self.y.append(y)

        y_pred = self.predict_by_all_memories(x, y)

        self.STM_X.append(x)
        self.STM_y.append(y)
        self.trainStepCount += 1

        self._adaptSTM()
        self._cleanLTM(x, y)
        self._enforceMaxLTMSize()
        self._enforceMaxSTMSize()

        self.ModelError = np.sqrt(np.mean(np.square(self.ModelPredHistory)))
        self.STMError = np.sqrt(np.mean(np.square(self.STMPredHistory)))
        self.LTMError = np.sqrt(np.mean(np.square(self.LTMPredHistory)))
        self.CMError = np.sqrt(np.mean(np.square(self.CMPredHistory)))
        self.STM_sizes.append(len(self.STM_X))
        self.LTM_sizes.append(len(self.LTM_X))

        return y_pred


    def partial_fit(self, X, y):

        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self._partial_fit(X[i,:], y[i]))

        return y_pred


    def getEuclideanDistances(self, sample, samples):

        return np.sqrt(libNearestNeighbor.get1ToNDistances(sample, samples))



    def getDistances(self, sample, samples):
        if self.metric_init and np.size(samples) != 0:
            if self.metric_smoothing:
                if self.metric_reducing:
                    idx = self.reducing_indices[-1]
                    reduced_matrix = self.metric_history_reduced[-1]
                    reduced_samples = np.array(samples)[:, idx]
                    reduced_sample = np.array(sample)[idx]
                    comp = sqrtm(reduced_matrix)
                    samples = reduced_samples.dot(comp.T)
                    sample = reduced_sample.dot(comp.T)
                else:
                    smoothed_matrix = self.metric_history_smoothed[-1]
                    comp = sqrtm(smoothed_matrix)
                    samples = np.array(samples).dot(comp.T)
                    sample = np.array(sample).dot(comp.T)
            else:
                if self.metric_reducing:
                    idx = self.reducing_indices[-1]
                    reduced_matrix = self.metric_history_reduced[-1]
                    reduced_samples = np.array(samples)[:, idx]
                    reduced_sample = np.array(sample)[idx]
                    comp = sqrtm(reduced_matrix)
                    samples = reduced_samples.dot(comp.T)
                    sample = reduced_sample.dot(comp.T)
                else:
                    samples = self.metric_learner.transform(samples)
                    sample = self.metric_learner.transform(np.stack([sample]))
                    sample = sample[0]

        return np.sqrt(libNearestNeighbor.get1ToNDistances(sample, samples))


    def getMetricDistances(self, sample, samples, samples_labels):
        try:
            if self.use_MLKR:
                if self.trainStepCount >= self.metric_step:
                    if self.trainStepCount % self.metric_step == 0:
                        logging.debug('Relearn metric after %d steps' % self.trainStepCount)
                        try:
                            self.metric_learner.fit(np.array(samples), np.array(samples_labels).squeeze())
                            logging.debug('New Metric:')
                            logging.debug(self.metric_learner.get_mahalanobis_matrix().diagonal())
                            if self.metric_smoothing:
                                logging.info('Metric Smoothing')
                                self.metric_history.insert(0,self.metric_learner.get_mahalanobis_matrix())
                                delta = 0.9
                                c = min(self.metric_smoothing_window, len(self.metric_history))
                                smoothed_matrix = delta * self.metric_history[c-1]
                                step = 2
                                for i in range(c-step,-1,-1):
                                    smoothed_matrix = np.add(smoothed_matrix, pow(delta, step)*self.metric_history[i])
                                    step+=1
                                smoothed_matrix = smoothed_matrix/c
                                self.metric_history_smoothed.append(smoothed_matrix)
                                self.relevancies.append(smoothed_matrix.diagonal())
                                max_rel = np.max(smoothed_matrix.diagonal())
                                self.relevancies_normalized.append(smoothed_matrix.diagonal() / max_rel)
                                self.relevancy_times.append(self.trainStepCount)
                                logging.debug('New Smoothed Metric:')
                                logging.debug(smoothed_matrix.diagonal())
                                if self.metric_reducing:
                                    if self.reducing_dimension == None:
                                        max_relevance = np.max(self.relevancies[-1])
                                        min_relevance = np.min(self.relevancies[-1])
                                        relevance_dist = max_relevance - min_relevance
                                        threshold_value = max_relevance - ((1 - self.relevance_threshold) * relevance_dist)
                                        idx = np.argwhere(self.relevancies[-1] > threshold_value).squeeze()
                                    else:
                                        idx = np.argpartition(self.relevancies[-1], -self.reducing_dimension)[-self.reducing_dimension:]
                                    if idx.size == 1:
                                        idx = idx[np.newaxis]
                                    reduced_matrix = smoothed_matrix[idx, :][:, idx]
                                    self.reducing_indices.append(idx)
                                    self.metric_history_reduced.append(reduced_matrix)
                                    self.relevancies_reduced.append(reduced_matrix.diagonal())
                                    reduced_samples = np.array(samples)[:, idx]
                                    reduced_sample = np.array(sample)[idx]
                                    comp = sqrtm(reduced_matrix)
                                    samples = reduced_samples.dot(comp.T)
                                    sample = reduced_sample.dot(comp.T)
                                    logging.debug('New Reduced Metric:')
                                    logging.debug(reduced_matrix.diagonal())
                                else:
                                    comp = sqrtm(smoothed_matrix)
                                    samples = np.array(samples).dot(comp.T)
                                    sample = np.array(sample).dot(comp.T)
                            else:
                                self.metric_history.insert(0, self.metric_learner.get_mahalanobis_matrix())
                                self.relevancies.append(self.metric_learner.get_mahalanobis_matrix().diagonal())
                                max_rel = np.max(self.metric_learner.get_mahalanobis_matrix().diagonal())
                                self.relevancies_normalized.append(self.metric_learner.get_mahalanobis_matrix().diagonal() / max_rel)
                                self.relevancy_times.append(self.trainStepCount)
                                if self.metric_reducing:
                                    if self.reducing_dimension == None:
                                        max_relevance = np.max(self.relevancies[-1])
                                        min_relevance = np.min(self.relevancies[-1])
                                        relevance_dist = max_relevance - min_relevance
                                        threshold_value = max_relevance - ((1 - self.relevance_threshold) * relevance_dist)
                                        idx = np.argwhere(self.relevancies[-1] > threshold_value).squeeze()
                                    else:
                                        idx = np.argpartition(self.relevancies[-1], -self.reducing_dimension)[-self.reducing_dimension:]
                                    if idx.size == 1:
                                        idx = idx[np.newaxis]
                                    reduced_matrix = self.metric_learner.get_mahalanobis_matrix()[idx, :][:, idx]
                                    self.reducing_indices.append(idx)
                                    self.metric_history_reduced.append(reduced_matrix)
                                    self.relevancies_reduced.append(reduced_matrix.diagonal())
                                    reduced_samples = np.array(samples)[:, idx]
                                    reduced_sample = np.array(sample)[idx]
                                    comp = sqrtm(reduced_matrix)
                                    samples = reduced_samples.dot(comp.T)
                                    sample = reduced_sample.dot(comp.T)
                                    logging.debug('New Reduced Metric:')
                                    logging.debug(reduced_matrix.diagonal())
                                else:
                                    samples = self.metric_learner.transform(samples)
                                    sample = self.metric_learner.transform(np.stack([sample]))
                                    sample = sample[0]
                            self.metric_init = True
                        except TypeError as e:
                            logging.info('Error in computing the MLKR metric. Using Euclidean distance for now.' + str(e))
                            return np.sqrt(libNearestNeighbor.get1ToNDistances(sample, samples))
                    elif self.metric_init:
                        if self.metric_smoothing:
                            if self.metric_reducing:
                                idx = self.reducing_indices[-1]
                                reduced_matrix = self.metric_history_reduced[-1]
                                reduced_samples = np.array(samples)[:, idx]
                                reduced_sample = np.array(sample)[idx]
                                comp = sqrtm(reduced_matrix)
                                samples = reduced_samples.dot(comp.T)
                                sample = reduced_sample.dot(comp.T)
                            else:
                                smoothed_matrix = self.metric_history_smoothed[-1]
                                comp = sqrtm(smoothed_matrix)
                                samples = np.array(samples).dot(comp.T)
                                sample = np.array(sample).dot(comp.T)
                        else:
                            if self.metric_reducing:
                                idx = self.reducing_indices[-1]
                                reduced_matrix = self.metric_history_reduced[-1]
                                reduced_samples = np.array(samples)[:, idx]
                                reduced_sample = np.array(sample)[idx]
                                comp = sqrtm(reduced_matrix)
                                samples = reduced_samples.dot(comp.T)
                                sample = reduced_sample.dot(comp.T)
                            else:
                                samples = self.metric_learner.transform(samples)
                                sample = self.metric_learner.transform(np.stack([sample]))
                                sample = sample[0]

        except ValueError as e:
            logging.info("Not enough data accumulated yet. Skipping." + str(e))
            return np.sqrt(libNearestNeighbor.get1ToNDistances(sample, samples))

        return np.sqrt(libNearestNeighbor.get1ToNDistances(sample, samples))


    def _predict(self, dist, idx, memory_y):

        memory_y = np.array(memory_y)
        idx = idx.squeeze()
        dist = dist[idx]
        memory_y = memory_y[idx]
        dist = dist + 0.00001

        weights = 1 / dist
        pred = np.matmul(weights, memory_y) / np.sum(weights)

        return pred


    def predict_by_all_memories(self, x, y):

        if len(self.STM_X) >= self.k:
            STM_dist = self.getMetricDistances(x, self.STM_X, self.STM_y)
            STM_idx = libNearestNeighbor.nArgMin(self.k, STM_dist)
            STM_pred = self._predict(STM_dist, STM_idx, self.STM_y)
        else:
            STM_pred = 0

        if len(self.LTM_X) >= self.k:
            LTM_dist = self.getDistances(x, self.LTM_X)
            LTM_idx = libNearestNeighbor.nArgMin(self.k, LTM_dist)
            LTM_pred = self._predict(LTM_dist, LTM_idx, self.LTM_y)
        else:
            LTM_pred = STM_pred

        if len(self.STM_X) + len(self.LTM_X) >= self.k:
            CM_dist = self.getDistances(x, self.STM_X + self.LTM_X)
            CM_idx = libNearestNeighbor.nArgMin(self.k, CM_dist)
            CM_pred = self._predict(CM_dist, CM_idx, self.STM_y + self.LTM_y)
        else:
            CM_pred = STM_pred

        #STM_len = len(self.STM_X)
        STM_len = 5
        preds = [STM_pred, LTM_pred, CM_pred]
        STM_weight = np.sqrt(np.mean(np.square(self.STMPredHistory[-STM_len:])))
        LTM_weight = np.sqrt(np.mean(np.square(self.LTMPredHistory[-STM_len:])))
        CM_weight = np.sqrt(np.mean(np.square(self.CMPredHistory[-STM_len:])))
        regressorChoice = np.argmin([STM_weight, LTM_weight, CM_weight])
        y_pred = preds[regressorChoice]

        self.regressorChoice.append(regressorChoice)
        self.STMPredHistory.append(STM_pred - y)
        self.LTMPredHistory.append(LTM_pred - y)
        self.CMPredHistory.append(CM_pred - y)
        self.ModelPredHistory.append(y_pred - y)
        self.STMPredictions.append(STM_pred)
        self.LTMPredictions.append(LTM_pred)
        self.CMPredictions.append(CM_pred)
        self.ModelPredictions.append(y_pred)

        return y_pred


    def _adaptSTM(self):

        if self.flag:
            if self.trainStepCount >= self.flag_occurence + self.k + 1:
                STMX = np.array(self.STM_X)
                STMy = np.array(self.STM_y)
                self.STM_X = STMX[self.k + 1:, :].tolist()
                self.STM_y = STMy[self.k + 1:].tolist()
                self.flag = False

        STMX = np.array(self.STM_X)
        STMy = np.array(self.STM_y)
        len_STM = len(self.STM_X)
        residuals = []
        for n in range(self.k, len_STM):
            sample = STMX[-len_STM + n, :]
            memory_x = STMX[-len_STM:-len_STM + n, :]
            memory_y = STMy[-len_STM:-len_STM + n]
            #STM_dist = self.getDistances(sample, memory_x)
            STM_dist = self.getEuclideanDistances(sample, memory_x)
            STM_idx = libNearestNeighbor.nArgMin(self.k, STM_dist)
            STM_pred = self._predict(STM_dist, STM_idx, memory_y)
            residuals.append(STM_pred - STMy[-len_STM + n])

        best_STM_error = np.sqrt(np.mean(np.square(residuals)))
        best_STM_size = STMX.shape[0]
        old_error = best_STM_error
        old_size = best_STM_size
        slice_size = int(STMX.shape[0] / 2)
        #slice_size = 40

        while (slice_size >= 50):
            residuals = []
            for n in range(self.k, slice_size):
                sample = STMX[-slice_size + n, :]
                memory_x = STMX[-slice_size:-slice_size + n, :]
                memory_y = STMy[-slice_size:-slice_size + n]
                #STM_dist = self.getDistances(sample, memory_x)
                STM_dist = self.getEuclideanDistances(sample, memory_x)
                STM_idx = libNearestNeighbor.nArgMin(self.k, STM_dist)
                STM_pred = self._predict(STM_dist, STM_idx, memory_y)
                residuals.append(STM_pred - STMy[-len_STM + n])

            STM_error = np.sqrt(np.mean(np.square(residuals)))

            if (STM_error < best_STM_error):
                best_STM_error = STM_error
                best_STM_size = slice_size

            slice_size = int(slice_size / 2)

        if (old_size != best_STM_size):
            self.STM_adaptions += 1
            self.STM_adaptions_idx.append(self.trainStepCount)

            print("ADAPTING: old size & error: ", old_size, old_error, "new size & error: ", best_STM_size, best_STM_error)

            discarded_X = STMX[0:-best_STM_size, :]
            discarded_y = STMy[0:-best_STM_size]
            #self.STM_X = STMX[-best_STM_size:, :].tolist()
            #self.STM_y = STMy[-best_STM_size:].tolist()
            self.STM_X = STMX[-(self.k + 1):, :].tolist()
            self.STM_y = STMy[-(self.k + 1):].tolist()
            self.flag = True
            self.flag_occurence = self.trainStepCount

            original_discard_size = len(discarded_X)

            discarded_X, discarded_y = self._cleanDiscarded(discarded_X, discarded_y)

            if (discarded_X.size):
                self.LTM_X += discarded_X.tolist()
                self.LTM_y += discarded_y.tolist()
                print("Added", len(discarded_X), "of", original_discard_size, "to LTM. ")
            else:
                print("All discarded Samples are dirty")


    def _clean_metric(self, diffs, dists, norm):

        return np.abs(diffs) * 1 / np.exp(dists / (norm))


    def _cleanDiscarded(self, discarded_X, discarded_y):

        STMX = np.array(self.STM_X)
        STMy = np.array(self.STM_y)
        clean_mask = np.zeros(discarded_X.shape[0], dtype=bool)
        for i in range(len(self.STM_X)):
            sample_x = STMX[i, :]
            sample_y = STMy[i]
            #STM_dist = self.getDistances(sample_x, STMX).squeeze()
            STM_dist = self.getEuclideanDistances(sample_x, STMX).squeeze()
            STM_idx = libNearestNeighbor.nArgMin(self.k + 1, STM_dist).squeeze()
            STM_dist = STM_dist[STM_idx]

            STM_dist_max = np.amax(STM_dist)
            STM_weighted_diff = self._clean_metric(STMy[STM_idx].squeeze() - sample_y, STM_dist, STM_dist_max)
            STM_weighted_diff_max = np.amax(STM_weighted_diff)

            #Discarded_dist = self.getDistances(sample_x, discarded_X)
            Discarded_dist = self.getEuclideanDistances(sample_x, discarded_X)
            Discarded_idx = np.where(Discarded_dist < STM_dist_max)[0]
            Discarded_dist = Discarded_dist[Discarded_idx]
            Discarded_weighted_diff = self._clean_metric(discarded_y[Discarded_idx].squeeze() - sample_y, Discarded_dist, STM_dist_max)

            clean_samples_idx = Discarded_idx[Discarded_weighted_diff < STM_weighted_diff_max]
            clean_mask[clean_samples_idx] = True

        cleaned_discarded_X = discarded_X[clean_mask]
        cleaned_discarded_y = discarded_y[clean_mask]

        return cleaned_discarded_X, cleaned_discarded_y


    def _cleanLTM(self, x, y):

        if len(self.LTM_X) >= self.k and len(self.STM_X) >= self.k:
            LTMX = np.array(self.LTM_X)
            LTMy = np.array(self.LTM_y)
            STMX = np.array(self.STM_X)
            STMy = np.array(self.STM_y)

            STM_dist = self.getDistances(x, STMX).squeeze()
            STM_idx = libNearestNeighbor.nArgMin(self.k + 1, STM_dist).squeeze()
            STM_dist = STM_dist[STM_idx]

            STM_dist_max = np.amax(STM_dist)
            STM_weighted_diff = self._clean_metric(STMy[STM_idx].squeeze() - y, STM_dist, STM_dist_max)
            STM_weighted_diff_max = np.amax(STM_weighted_diff)

            LTM_dist = self.getDistances(x, LTMX).squeeze()
            LTM_idx = np.where(LTM_dist < STM_dist_max)[0]
            LTM_dist = LTM_dist[LTM_idx]
            LTM_weighted_diff = self._clean_metric(LTMy[LTM_idx].squeeze() - y, LTM_dist, STM_dist_max)

            dirty_samples_idx = LTM_idx[LTM_weighted_diff > STM_weighted_diff_max]

            if dirty_samples_idx.shape[0] > 1:
                self.LTM_X = np.delete(LTMX, dirty_samples_idx, axis=0).tolist()
                self.LTM_y = np.delete(LTMy, dirty_samples_idx, axis=0).tolist()


    def _enforceMaxLTMSize(self):

        while (len(self.LTM_X) > self.max_LTM_size):
            LTM_dist = np.sqrt(libNearestNeighbor.getNToNDistances(self.LTM_X, self.LTM_X))
            for i in range(len(self.LTM_X)):
                #LTM_dist[i][i] += 10000
                LTM_dist[i][i] = np.inf
            LTM_dist_min_idx_1 = np.argmin(np.min(LTM_dist, axis=0))
            LTM_dist_min_idx_2 = np.argmin(LTM_dist[LTM_dist_min_idx_1])
            closest_point_1_x = self.LTM_X[LTM_dist_min_idx_1]
            closest_point_1_y = self.LTM_y[LTM_dist_min_idx_1]
            closest_point_2_x = self.LTM_X[LTM_dist_min_idx_2]
            closest_point_2_y = self.LTM_y[LTM_dist_min_idx_2]
            merged_point_x = np.average([closest_point_1_x, closest_point_2_x], axis=0)
            merged_point_y = np.average([closest_point_1_y, closest_point_2_y], axis=0)
            self.LTM_X = np.delete(np.array(self.LTM_X), [LTM_dist_min_idx_1, LTM_dist_min_idx_2], axis=0).tolist()
            self.LTM_y = np.delete(np.array(self.LTM_y), [LTM_dist_min_idx_1, LTM_dist_min_idx_2], axis=0).tolist()
            self.LTM_X.append(merged_point_x)
            self.LTM_y.append(merged_point_y)


    def _enforceMaxSTMSize(self):

        if len(self.STM_X) > self.max_STM_size:
            self.STM_X.pop(0)
            self.STM_y.pop(0)






    def plot_relevancies(self):

        rel = np.array(self.relevancies)
        x = np.arange(len(self.relevancies))
        fig = plt.figure()
        for i in range(self.relevancies[0].shape[0]):
            plt.plot(self.relevancy_times, rel[:, i], label='Feature ' + str(i))
        plt.xlabel('Time')
        plt.ylabel('Relevance')
        plt.title('Feature Relevancies over time', fontweight='bold')
        plt.legend()
        plt.show()


    def plot_relevancies_normalized(self):

        rel = np.array(self.relevancies_normalized)
        x = np.arange(len(self.relevancies_normalized))
        fig = plt.figure()
        for i in range(self.relevancies_normalized[0].shape[0]):
            plt.plot(self.relevancy_times, rel[:, i], label='Feature ' + str(i))
        plt.xlabel('Time')
        plt.ylabel('Normalized Relevance')
        plt.title('Normalized Feature Relevancies over time', fontweight='bold')
        plt.legend()
        plt.show()


    def plot_summed_relevancies(self, start=None, end=None, mean=False):

        rel = np.array(self.relevancies)
        if (start == None):
            if (mean == True):
                rel_sum = np.mean(rel, axis=0)
                title = "Average Relevance for all Features"
            else:
                rel_sum = np.sum(rel, axis=0)
                title = "Summed Relevance for all Features"
        else:
            if (mean == True):
                rel_sum = np.mean(rel[start:end,:], axis=0)
                title = "Average Relevance for all Features for Interval [" + str(start) + "," + str(end) + "]"
            else:
                rel_sum = np.sum(rel[start:end,:], axis=0)
                title = "Summed Relevance for all Features for Interval [" + str(start) + "," + str(end) + "]"
        x = np.arange(rel.shape[1])
        fig = plt.figure()
        plt.bar(x, rel_sum)
        plt.title(title)
        plt.xlabel('Feautures')
        plt.ylabel('Relevance')
        plt.show()


    def plot_summed_relevancies_normalized(self, start=None, end=None, mean=False):

        #rel = np.array(self.relevancies_normalized)
        rel = np.array([x / np.sum(x) for x in self.relevancies])
        if (start == None):
            if (mean == True):
                rel_sum = np.mean(rel, axis=0)
                title = "Average Normalized Relevance for all Features"
            else:
                rel_sum = np.sum(rel, axis=0)
                title = "Summed Normalized Relevance for all Features"
        else:
            if (mean == True):
                rel_sum = np.mean(rel[start:end, :], axis=0)
                title = "Average Normalized Relevance for all Features for Interval [" + str(start) + "," + str(end) + "]"
            else:
                rel_sum = np.sum(rel[start:end, :], axis=0)
                title = "Summed Normalized Relevance for all Features for Interval [" + str(start) + "," + str(end) + "]"
        x = np.arange(rel.shape[1])
        fig = plt.figure()
        plt.bar(x, rel_sum)
        plt.title(title)
        plt.xlabel('Feautures')
        plt.ylabel('Relevance')
        plt.show()


    def plot_memory_sizes(self):

        fig, axs = plt.subplots(2, sharex='all')
        x = np.arange(len(self.STM_sizes))
        axs[0].plot(x, self.regressorChoice, label='Regressor Choice')
        axs[0].set_yticks([0, 1, 2])
        axs[0].set_yticklabels(['STM', 'LTM', 'CM'])
        axs[1].plot(x, self.STM_sizes, label='STM')
        axs[1].plot(x, self.LTM_sizes, label='LTM')
        for i in range(len(self.STM_adaptions_idx)):
            plt.axvline(x=self.STM_adaptions_idx[i], color='black', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Size')
        plt.title('STM and LTM sizes. Number of STM adaptions = ' + str(self.STM_adaptions))
        plt.legend()
        plt.show()


    def plot_predictions(self):

        fig = plt.figure()
        plt.plot(self.y, label='Ground Truth', alpha=0.5)
        plt.plot(self.STMPredictions, label='STM', alpha=0.5)
        plt.plot(self.LTMPredictions, label='LTM', alpha=0.5)
        plt.plot(self.CMPredictions, label='CM', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Prediction')
        plt.title('STM_E = ' + str(self.STMError) + '  LTM_E = ' + str(self.LTMError) + '  CM_E = ' + str(self.CMError) + '  Model_E = ' + str(self.ModelError))
        plt.legend()
        plt.show()











































