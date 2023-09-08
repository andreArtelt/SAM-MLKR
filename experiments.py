import numpy as np
import logging
import pickle
from SAM_MLKR import SAM_MLKR




# load data from disk
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)




X = data['X']
y = [data['y_one_dim'], data['y_two_dim'], data['y_all_dim'], data['y_continuous']]
max_STM_sizes = [50, 250, 500]

models_ = [[], [], [], []]
errors_ = [[], [], [], []]
for i in range(len(y)):

    STM_models = [[], [], []]
    STM_errors = [[], [], []]
    for j in range(len(max_STM_sizes)):

        models = [[], [], [], [], []]
        errors = [[], [], [], [], []]
        for k in range(50):

            model_1 = SAM_MLKR(k=5, max_STM_size=max_STM_sizes[j], max_LTM_size=500, use_MLKR=False, metric_step=100, metric_smoothing=False, metric_reducing=False, relevance_threshold=0.5)
            model_1.partial_fit(X, y[i])

            model_2 = SAM_MLKR(k=5, max_STM_size=max_STM_sizes[j], max_LTM_size=500, use_MLKR=True, metric_step=100, metric_smoothing=False, metric_reducing=False, relevance_threshold=0.5)
            model_2.partial_fit(X, y[i])

            model_3 = SAM_MLKR(k=5, max_STM_size=max_STM_sizes[j], max_LTM_size=500, use_MLKR=True, metric_step=100, metric_smoothing=True, metric_reducing=False, relevance_threshold=0.5)
            model_3.partial_fit(X, y[i])

            model_4 = SAM_MLKR(k=5, max_STM_size=max_STM_sizes[j], max_LTM_size=500, use_MLKR=True, metric_step=100, metric_smoothing=False, metric_reducing=True, relevance_threshold=0.5)
            model_4.partial_fit(X, y[i])

            model_5 = SAM_MLKR(k=5, max_STM_size=max_STM_sizes[j], max_LTM_size=500, use_MLKR=True, metric_step=100, metric_smoothing=True, metric_reducing=True, relevance_threshold=0.5)
            model_5.partial_fit(X, y[i])


            m_list = [model_1, model_2, model_3, model_4, model_5]

            for l in range(5):
                models[l].append(m_list[l])
                errors[l].append(m_list[l].ModelError)

        STM_models[j] = models
        STM_errors[j] = errors

    models_[i] = STM_models
    errors_[i] = STM_errors

# save results and models
with open('models.pkl', 'wb') as file:
    pickle.dump(models_, file)

with open('errors.pkl', 'wb') as file:
    pickle.dump(errors_, file)




# print results for experiment

print('one_dim_results:')

print('STM_size: 50')
print(np.mean(errors_[0][0][0]))
print(np.mean(errors_[0][0][1]))
print(np.mean(errors_[0][0][2]))
print(np.mean(errors_[0][0][3]))
print(np.mean(errors_[0][0][4]))

print('############################')

print('STM_size: 250')
print(np.mean(errors_[0][1][0]))
print(np.mean(errors_[0][1][1]))
print(np.mean(errors_[0][1][2]))
print(np.mean(errors_[0][1][3]))
print(np.mean(errors_[0][1][4]))

print('############################')

print('STM_size: 500')
print(np.mean(errors_[0][2][0]))
print(np.mean(errors_[0][2][1]))
print(np.mean(errors_[0][2][2]))
print(np.mean(errors_[0][2][3]))
print(np.mean(errors_[0][2][4]))

print('############################')
print('############################')
print('############################')

print('two_dim_results:')

print('STM_size: 50')
print(np.mean(errors_[1][0][0]))
print(np.mean(errors_[1][0][1]))
print(np.mean(errors_[1][0][2]))
print(np.mean(errors_[1][0][3]))
print(np.mean(errors_[1][0][4]))

print('############################')

print('STM_size: 250')
print(np.mean(errors_[1][1][0]))
print(np.mean(errors_[1][1][1]))
print(np.mean(errors_[1][1][2]))
print(np.mean(errors_[1][1][3]))
print(np.mean(errors_[1][1][4]))

print('############################')

print('STM_size: 500')
print(np.mean(errors_[1][2][0]))
print(np.mean(errors_[1][2][1]))
print(np.mean(errors_[1][2][2]))
print(np.mean(errors_[1][2][3]))
print(np.mean(errors_[1][2][4]))

print('############################')
print('############################')
print('############################')

print('all_dim_results:')

print('STM_size: 50')
print(np.mean(errors_[2][0][0]))
print(np.mean(errors_[2][0][1]))
print(np.mean(errors_[2][0][2]))
print(np.mean(errors_[2][0][3]))
print(np.mean(errors_[2][0][4]))

print('############################')

print('STM_size: 250')
print(np.mean(errors_[2][1][0]))
print(np.mean(errors_[2][1][1]))
print(np.mean(errors_[2][1][2]))
print(np.mean(errors_[2][1][3]))
print(np.mean(errors_[2][1][4]))

print('############################')

print('STM_size: 500')
print(np.mean(errors_[2][2][0]))
print(np.mean(errors_[2][2][1]))
print(np.mean(errors_[2][2][2]))
print(np.mean(errors_[2][2][3]))
print(np.mean(errors_[2][2][4]))

print('############################')
print('############################')
print('############################')

print('continuous_results:')

print('STM_size: 50')
print(np.mean(errors_[3][0][0]))
print(np.mean(errors_[3][0][1]))
print(np.mean(errors_[3][0][2]))
print(np.mean(errors_[3][0][3]))
print(np.mean(errors_[3][0][4]))

print('############################')

print('STM_size: 250')
print(np.mean(errors_[3][1][0]))
print(np.mean(errors_[3][1][1]))
print(np.mean(errors_[3][1][2]))
print(np.mean(errors_[3][1][3]))
print(np.mean(errors_[3][1][4]))

print('############################')

print('STM_size: 500')
print(np.mean(errors_[3][2][0]))
print(np.mean(errors_[3][2][1]))
print(np.mean(errors_[3][2][2]))
print(np.mean(errors_[3][2][3]))
print(np.mean(errors_[3][2][4]))










