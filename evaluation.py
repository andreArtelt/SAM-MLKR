import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pickle

base_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# load results from disk

with open('models.pkl', 'rb') as file:
    models = pickle.load(file)
with open('errors.pkl', 'rb') as file:
    errors = pickle.load(file)




# define title words
dataset_names = ['one_dim', 'two_dim', 'all_dim', 'continuous']
stm_lengths = ['50', '250', '500']
model_names = ['SAM-Knn', 'SAM-M', 'SAM-M-S', 'SAM-M-R', 'SAM-M-SR']



# choose dataset and stm length
dataset_idx = 0
stm_length_idx = 0
model_idx = 1

def plot_relevancies(dataset_idx, stm_length_idx, model_idx, normalized_relevancies=False):

    list = models[dataset_idx][stm_length_idx][model_idx]
    if (normalized_relevancies == True):
        relevancies = [np.array(x.relevancies_normalized) for x in list]
        title = 'Normalized Feature Relevancies, '+ dataset_names[dataset_idx] + ', Max_STM = ' + str(list[0].max_STM_size)
    else:
        relevancies = [np.array(x.relevancies) for x in list]
        title = 'Feature Relevancies, '+ dataset_names[dataset_idx] + ', Max_STM = ' + str(list[0].max_STM_size)
    relevancy_times = [x.relevancy_times for x in list][0]

    mean_array = np.zeros(relevancies[0].shape)
    std_array = np.zeros(relevancies[0].shape)
    for i in range(mean_array.shape[0]):
        for j in range(mean_array.shape[1]):
            values = [x[i][j] for x in relevancies]
            mean_array[i][j] = np.mean(values)
            std_array[i][j] = np.std(values)

    sns.set()
    fig = plt.figure()
    for i in range(mean_array.shape[1]):
        plt.plot(relevancy_times, mean_array[:,i], label='Feature ' + str(i))
        plt.fill_between(relevancy_times, mean_array[:,i] - std_array[:,i], mean_array[:,i] + std_array[:,i], alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel('Relevance')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.show()


def barplot_errors(dataset_idx, stm_length_idx):

    mean_errors = []
    std_errors = []
    for i in range(5):
        mean_errors.append(np.mean(errors[dataset_idx][stm_length_idx][i]))
        std_errors.append(np.std(errors[dataset_idx][stm_length_idx][i]))

    list = [np.array(errors[dataset_idx][stm_length_idx][i]) for i in range(5)]

    sns.set()
    fig, ax = plt.subplots()
    ax.boxplot(list)
    ax.set_title('Model Error, ' + dataset_names[dataset_idx] + ', ' + stm_lengths[stm_length_idx])
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_xlabel('Models')
    ax.set_ylabel('Error')
    plt.show()


def plot_errors_all_STMs(dataset_idx):

    e50 = []
    e250 = []
    e500 = []
    for i in range(5):
        e50.append(errors[dataset_idx][0][i][0])
        e250.append(errors[dataset_idx][1][i][0])
        e500.append(errors[dataset_idx][2][i][0])

    x = np.arange(5)

    fig = plt.figure()
    plt.plot(x, e50, 'o-', label='STM_Max = 50')
    plt.plot(x, e250, 'o-', label='STM_Max = 250')
    plt.plot(x, e500, 'o-', label='STM_Max = 500')
    plt.legend()
    plt.title('Model Errors for all STM sizes, ' + dataset_names[dataset_idx], fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.xticks(x, model_names)
    plt.show()


def plot_errors_all_STMs_for_multiple_parameters(dataset_idx):

    e50_1 = []
    e250_1 = []
    e500_1 = []
    for i in range(5):
        e50_1.append(errors_10000[dataset_idx][0][i][0])
        e250_1.append(errors_10000[dataset_idx][1][i][0])
        e500_1.append(errors_10000[dataset_idx][2][i][0])

    e50_2 = []
    e250_2 = []
    e500_2 = []
    for i in range(5):
        e50_2.append(errors_50000[dataset_idx][0][i][0])
        e250_2.append(errors_50000[dataset_idx][1][i][0])
        e500_2.append(errors_50000[dataset_idx][2][i][0])

    x = np.arange(5)

    sns.set()
    fig = plt.figure()
    plt.plot(x, e50_1, 'o-', color=base_colors['darkblue'], label='STM_Max = 50, gamma = 0.5')
    plt.plot(x, e250_1, 'o-', color=base_colors['darkorange'], label='STM_Max = 250, gamma = 0.5')
    plt.plot(x, e500_1, 'o-', color=base_colors['mediumvioletred'], label='STM_Max = 500, gamma = 0.5')
    plt.plot(x, e50_2, 'o-', color=base_colors['cornflowerblue'], label='STM_Max = 50, gamma = 2')
    plt.plot(x, e250_2, 'o-', color=base_colors['bisque'], label='STM_Max = 250, gamma = 2')
    plt.plot(x, e500_2, 'o-', color=base_colors['palevioletred'], label='STM_Max = 500, gamma = 2')
    plt.legend()
    plt.title('Model Errors for all STM sizes, ' + dataset_names[dataset_idx], fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.xticks(x, model_names)
    plt.show()



def plot_reduced_features_over_time(dataset_idx, stm_length_idx, model_idx):

    model = models[dataset_idx][stm_length_idx][model_idx][0]
    ind = model.reducing_indices
    times = model.relevancy_times
    data_length = model.trainStepCount

    x = [[], [], [], [], [], [], [], [], [], []]
    list = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(ind)):
        for j in range(10):
            if len(ind[i]) > j:
                list[j].append(ind[i][j])
                x[j].append(times[i])

    for i in range(10):
        if len(x[i]) == 0:
            x = x[0:i]
            list = list[0:i]
            break

    fig = plt.figure()
    for i in range(len(x)):
        plt.scatter(x[i], list[i])
    plt.ylabel('Feature')
    plt.xlabel('Time')
    plt.yticks(np.arange(10))
    plt.xticks(np.arange(10) * 1000)
    plt.title('Reduced Features, '+ dataset_names[dataset_idx] + ', Max_STM = ' + stm_lengths[stm_length_idx] + ', ' + model_names[model_idx], fontweight='bold')
    plt.show()





#for i in range(4):
#    plot_errors_all_STMs(i)

#for i in range(3):
#    plot_reduced_features_over_time(2, i, 3)
#    plot_reduced_features_over_time(2, i, 4)




#models = models_50000
#errors = errors_50000

#for i in range(4):
    #plot_errors_all_STMs(i)
    #plot_errors_all_STMs_for_multiple_parameters(i)


#for i in range(3):
#    plot_reduced_features_over_time(2, i, 3)
#    plot_reduced_features_over_time(2, i, 4)


#for i in range(3):
#    for j in range(4):
#        plot_relevancies(3,i,j+1)

#plot_reduced_features_over_time(2,0,4)


plot_relevancies(0, 2, 1, normalized_relevancies=True)
plot_relevancies(0,2,2, normalized_relevancies=True)
plot_relevancies(1, 2, 1, normalized_relevancies=True)
plot_relevancies(1,2,2, normalized_relevancies=True)
plot_relevancies(2, 2, 1, normalized_relevancies=True)
plot_relevancies(2,2,2, normalized_relevancies=True)
plot_relevancies(3, 2, 1, normalized_relevancies=True)
plot_relevancies(3,2,2, normalized_relevancies=True)







#plot_relevancies(3,0,1, normalized_relevancies=True)









