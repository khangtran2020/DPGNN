import numpy as np
from DataProcessing import sigmoid, binary_to_float, float_to_binary, join_string, string_to_int


def compute_overall_score(shap_matrix, feat_matrix, masked_matrix, gamma):
    num_data, num_feat = feat_matrix.shape
    shap_matrix = np.abs(shap_matrix)
    sens_matrix = np.abs(feat_matrix - masked_matrix)

    max_sens = np.tile(np.expand_dims(np.max(sens_matrix, axis=1), axis=1), (1, num_feat))
    min_sens = np.tile(np.expand_dims(np.min(sens_matrix, axis=1), axis=1), (1, num_feat))
    max_shap = np.tile(np.expand_dims(np.max(shap_matrix, axis=1), axis=1), (1, num_feat))
    min_shap = np.tile(np.expand_dims(np.min(shap_matrix, axis=1), axis=1), (1, num_feat))

    # inverse sens score
    inv_sens_matrix = max_sens - sens_matrix + min_sens

    # normalize
    sum_shap = np.tile(np.expand_dims(np.sum(shap_matrix, axis=1), axis=-1), (1, num_feat))
    shap_matrix = shap_matrix / sum_shap
    inv_sens_matrix = sens_matrix / inv_sens_matrix
    max_sens = np.tile(np.expand_dims(np.max(inv_sens_matrix, axis=1), axis=1), (1, num_feat))
    min_sens = np.tile(np.expand_dims(np.min(inv_sens_matrix, axis=1), axis=1), (1, num_feat))

    # scale to the same range
    inv_sens_matrix = (inv_sens_matrix - min_sens) / (max_sens - min_sens) * (max_shap - min_shap)

    # overall score
    overall_score = gamma * shap_matrix + (1 - gamma) * inv_sens_matrix
    return overall_score


def compute_mu_for_each_bit(eps, overall_score, num_bit, int_bit, use_sign_bit=True):
    num_data, num_feat = overall_score.shape
    sens = []
    if use_sign_bit:
        sens.append(2 ** (int_bit + 1))
        for i in range(1, num_bit):
            sens.append(2 ** (int_bit - i))
    else:
        for i in range(1, num_bit + 1):
            sens.append(2 ** (int_bit - i))
    sens = np.array(sens)
    sens = sens / (np.sum(sens))
    sens = np.tile(sens, (num_data, num_feat))
    overall_score = np.repeat(overall_score, num_bit, axis=1)
    mu = eps * overall_score * (num_feat - sens)
    return mu, overall_score


def FaRR(feat_matrix, masked_matrix, shap_matrix, gamma, num_bit, int_bit, eps, use_sign_bit):
    num_feat = feat_matrix.shape[1]
    # vectorize functions
    float_bin = lambda x: float_to_binary(x, int_bit, num_bit - int_bit - 1)
    float_to_binary_vec = np.vectorize(float_bin)
    bin_float = lambda x: binary_to_float(x, int_bit, num_bit - int_bit - 1)
    binary_to_float_vec = np.vectorize(bin_float)

    # get parameter
    overall_score = compute_overall_score(feat_matrix=feat_matrix, masked_matrix=masked_matrix, shap_matrix=shap_matrix,
                                          gamma=gamma)
    mu, overall_score = compute_mu_for_each_bit(eps=eps, overall_score=overall_score, num_bit=num_bit, int_bit=int_bit,
                                                use_sign_bit=use_sign_bit)

    # randomizing
    feat = float_to_binary_vec(feat_matrix)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat)
    p = sigmoid(num_feat * overall_score * eps - mu)
    del (mu)
    del (overall_score)
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    del (p)
    del (p_temp)
    perturb_feat = (perturb + feat) % 2
    del (perturb)
    del (feat)
    perturb_feat = np.apply_along_axis(join_string, 1, perturb_feat, num_bit, num_feat)
    perturb_feat = binary_to_float_vec(perturb_feat)
    return perturb_feat
