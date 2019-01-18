import numpy as np
import matplotlib.pyplot as plt
from approximation import gram_similarity, load_pickle

def create_plot():
    full_kernel = load_pickle("pickels/100_doc_gram_matrix")
    prefix_top_features = "pickels/approx_for_graph/gram_matrix_%d_top_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl"
    prefix_inv_features = "pickels/approx_for_graph/gram_matrix_%d_inv_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl"
    prefix_rnd_features = "pickels/approx_for_graph/gram_matrix_%d_rnd_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl"
    feature_range = [ x for x in range(5, 200, 5) ]
    # feature_range_2 = [ x for x in range(200, 10200, 200) ]
    feature_range_2 = [ x for x in range(200, 3000, 200) ]
    feature_range_3 = [ x for x in range(3000, 7000, 1000) ]
    feature_range.extend(feature_range_2)
    feature_range.extend(feature_range_3)
    feature_range.append(10000)

    # y_axis_one = []
    y_axis_top = []
    y_axis_inv = []
    y_axis_rnd = []
    for feat in feature_range:
        print("Calculating for %d" % feat)
        top_kernel = load_pickle(prefix_top_features % feat)
        inv_kernel = load_pickle(prefix_inv_features % feat)
        rnd_kernel = load_pickle(prefix_rnd_features % feat)
        y_axis_top.append(gram_similarity(full_kernel, top_kernel))
        y_axis_inv.append(gram_similarity(full_kernel, inv_kernel))
        y_axis_rnd.append(gram_similarity(full_kernel, rnd_kernel))
        # y_axis_one.append(1)
    
    # plt.plot(feature_range, y_axis_one)
    plt.plot(feature_range, y_axis_top, '-')
    plt.plot(feature_range, y_axis_inv, ':')
    plt.plot(feature_range, y_axis_rnd, '--')
    plt.axis([ 0, 10000, 0.5, 1.01])
    plt.legend(['Frequent','Infrequent', 'Random'], loc='lower right')
    plt.title("Matrix alignment")
    plt.xlabel("Number of features")
    plt.ylabel("Gram similarity")
    plt.show()

if __name__ == '__main__':
    create_plot()
