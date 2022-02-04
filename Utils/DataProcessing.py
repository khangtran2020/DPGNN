import numpy as np

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax function
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# normalize for shap score and sensitive score
def normalize(x, flip):
    if flip == True:
        x = np.max(x) - x + np.min(x)
        return softmax(x)
    else:
        return softmax(x)

# combining scores
def overal_score(alpha, beta, gamma):
    return gamma*alpha + (1-gamma)*beta

# float to binary
def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    if x >= 0:
        res = '0' + res
    else:
        res = '1' + res
    return res

# binary to float
def binary_to_float(bstr, m, n):
    sign = bstr[0]
    bs = bstr[1:]
    res = int(bs, 2) / 2 ** n
    if sign == 1:
        res = -1 * res
    return res

def list_edge_from_adj(adj_matrix):
    new_adj = np.array(adj_matrix > 0).astype(int)
    new_adj = new_adj - np.identity(new_adj.shape[0])
    left = np.where(new_adj == 1)[0]
    right = np.where(new_adj == 1)[1]
    list_edge = []
    for a in range(left.shape[0]):
        i = left[a]
        j = right[a]
        if ((j,i) not in list_edge) & ((i,j) not in list_edge):
            list_edge.append((i,j))
    return list_edge

def pick_private_edge(list_edge, p_rate):
    list_edge_dict = {}
    cnt = 0
    for e in list_edge:
        chose_private_edge = np.random.choice(2,1, p=[1-p_rate, p_rate])
        list_edge_dict[e] = chose_private_edge[0]
        if (chose_private_edge[0] == 1):
            cnt += 1
    return list_edge_dict, cnt

def laplacian_matrix(adj_matrix):
    new_adj = adj_matrix + np.identity(adj_matrix.shape[0])
    D = np.sum(new_adj, axis=1)
    D = np.diag(D)
    D = np.power(D,-0.5)
    D[D == np.inf] = 0
    adj = np.dot(np.dot(D,new_adj),D)
    return adj

def adj_from_edge_list(edge_list, num_node):
    new_adj = np.zeros(shape=(num_node,num_node))
    for e in edge_list:
        new_adj[e[0], e[1]] = 1
        new_adj[e[1], e[0]] = 1
    return new_adj

def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

def duchi_mechanism(x, eps):
    p_temp = (np.exp(eps)-1)/(2*np.exp(eps)+2)*x+1/2
    p = np.random.rand()
    if (p_temp - p > 0):
        return (np.exp(eps)+1)/(np.exp(eps)-1)
    else:
        return -1*(np.exp(eps)+1)/(np.exp(eps)-1)

def piecewise_mechanism(x, eps):
    z = np.exp(eps/2)
    p_1 = (x+1)/(2+2*z)
    p_2 = z/(z+1)
    p_3 = (1-x)/(2+2*z)
    C = (z+1)/(z-1)
    g_1 = (C+1)*x/2-(C-1)/2
    g_2 = (C+1)*x/2 + (C-1)/2
    p = np.random.rand()
    if (p < p_1):
        return -1*C + np.random.rand()*(g_1 + C)
    elif (p < p_1 + p_2):
        return (g_2 - g_1)*np.random.rand() + g_1
    else:
        return g_2 + np.random.rand()*(C-g_2)

def hybrid_mechanism(x, eps):
    if (eps <= 0.61):
        return duchi_mechanism(x, eps)
    else:
        p = np.random.rand()
    if (p <= np.exp(-eps/2)):
        return duchi_mechanism(x, eps)
    else:
        return piecewise_mechanism(x, eps)

def three_output_mechanism(x, eps):
    delta_0 = np.exp(4 * eps) + 14 * np.exp(3 * eps) + 50 * np.exp(2 * eps) - 2 * np.exp(eps) + 25
    delta_1 = -2 * np.exp(6 * eps) - 42 * np.exp(5 * eps) - 270 * np.exp(4 * eps) - 404 * np.exp(
        3 * eps) - 918 * np.exp(2 * eps) + 30 * np.exp(eps) - 250
    tmp = -1 / 6 * (-np.exp(2 * eps) - 4 * np.exp(eps) - 5 + 2 * np.sqrt(delta_0) * np.cos(np.pi / 3 + 1 / 3 * np.arccos(-1*delta_1 / (2 * delta_0 ** (3 / 2)))))
    if (eps < np.log(2)):
        a = 0
    elif (eps >= np.log(2) and eps < np.log(5.53)):
        a = tmp
    else:
        a = np.exp(eps) / (np.exp(eps) + 2)
    b = a * (np.exp(eps) - 1) / np.exp(eps)
    d = (a + 1) * (np.exp(eps) - 1) / (2 * (np.exp(eps) + 1))
    C = np.exp(eps) * (np.exp(eps) + 1) / ((np.exp(eps) - 1) * (np.exp(eps) - a))

    if (x >= 0):
        p_1 = (1 - a) / 2 - (-b + d) * x
    else:
        p_1 = (1 - a) / 2 - d * x
    p_2 = a - b * np.abs(x)
    if (x >= 0):
        p_3 = (1 - a) / 2 + d * x
    else:
        p_3 = (1 - a) / 2 + (-b + d) * x

    p = np.random.rand()
    if (p <= p_1):
        return float(-1 * C)
    elif (p <= p_2 + p_1):
        return 0.0
    else:
        return float(C)


def PM_SUB(x, eps):
    z = np.exp(2*eps / 3)
    t = np.exp(eps / 3)
    P1 = (x + 1)*t / (2*t + 2 * np.exp(eps))
    P2 = np.exp(eps)/(np.exp(eps) + t)
    P3 = (1 - x)*t / (2 * t + 2 * np.exp(eps))

    C = (np.exp(eps) + t)*(t + 1) / (t*(np.exp(eps) - 1))
    g1 = (np.exp(eps) + t)*(x*t - 1)/(t*(np.exp(eps) - 1))
    g2 = (np.exp(eps) + t)*(x*t + 1)/(t*(np.exp(eps) - 1))

    p = np.random.rand()

    if (p < P1):
        return -C + np.random.rand()*(g1 + C)
    elif (p < P1 + P2):
        return (g2 - g1)*np.random.rand() + g1
    else:
        return (C - g2)* + g2

def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise

def min_max_norm(x):
    r = x.shape[1]
    min_by_row = np.min(x, axis=1)
    range_by_row = np.max(x, axis=1) - np.min(x, axis=1)
    min_by_row = np.expand_dims(min_by_row, axis=-1)
    range_by_row = np.expand_dims(range_by_row, axis=-1)
    min_by_row = np.tile(min_by_row, (1, r))
    range_by_row = np.tile(range_by_row, (1, r))
    x = (x - min_by_row)/range_by_row
    return x