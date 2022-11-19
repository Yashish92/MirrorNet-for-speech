import numpy as np

def ppmc(x, y):
    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    num = np.nansum((x - xbar)*(y - ybar))
    den = np.sqrt(np.nansum((x - xbar)**2))*np.sqrt(np.nansum((y - ybar)**2))
    corr = num/den

    return corr

# computing average correlations on test set
def compute_corr_score(y_predict, y_true, no_TVs):
    No_TVs = no_TVs
    corr_TVs = np.zeros(No_TVs)
    corr_TVs_pc = np.zeros(No_TVs)
    tot_samples = y_true.shape[0]
    for j in range(0, y_true.shape[0]):
        for i in range(0, No_TVs):
            corr_TVs[i] += ppmc(y_predict[j,:,i], y_true[j, :, i])
            # corr, _ = pearsonr(y_predict[j,:,i], y_true[j, :, i])
            # corr_TVs_pc[i] += corr
    # for j in range(0, y_val.shape[0]):
    #     for i in range(0, No_TVs):
    #         corr_TVs[i] += ppmc(y_predict_val[j,:,i], y_val[j, :, i])
    corr_TVs_avg = corr_TVs/tot_samples
    avg_corr_tvs = np.mean(corr_TVs_avg)
    avg_corr_6TVs = np.mean(corr_TVs_avg[0:6])
    # corr_TVs_pc = corr_TVs_pc/tot_samples

    print("Corr_Average_Test_set :", corr_TVs_avg)
    print("Corr_Average_across_all:", avg_corr_tvs)
    print("Corr_Average_across_6TVs:", avg_corr_6TVs)
    # print("Corr_pearson_func :", corr_TVs_pc)

    return corr_TVs_avg, avg_corr_tvs, avg_corr_6TVs    # print("Corr_pearson_func :", corr_TVs_pc)
