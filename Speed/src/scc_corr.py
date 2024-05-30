import os

import pandas as pd
from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', 10)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
pd.set_option('expand_frame_repr', False)

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}

class Coordinate_cluster:
    def __init__(self, A):
        # lat = 纬度， lon = 经度
        # 添加所属聚类中心属性,总计红、绿、蓝
        self.a = str(A) + "_a"
        self.b = str(A) + "_b"
        self.c = str(A) + "_c"
        self.d = str(A) + "_d"
        self.e = str(A) + "_e"
        self.f = str(A) + "_f"
        self.g = str(A) + "_g"
        self.h = str(A) + '_h'
        self.i = str(A) + '_i'
        self.j = str(A) + '_j'

def QUBO_cluster(dict, config):
    cqm = ConstrainedQuadraticModel()

    for key, val in dict.items():
        x = Coordinate_cluster(key[0])
        y = Coordinate_cluster(key[1])

        x.a = Binary('{}'.format(x.a))
        x.b = Binary('{}'.format(x.b))
        x.c = Binary('{}'.format(x.c))
        x.d = Binary('{}'.format(x.d))
        x.e = Binary('{}'.format(x.e))
        x.f = Binary('{}'.format(x.f))
        x.g = Binary('{}'.format(x.g))
        x.h = Binary('{}'.format(x.h))
        x.i = Binary('{}'.format(x.i))
        x.j = Binary('{}'.format(x.j))

        y.a = Binary('{}'.format(y.a))
        y.b = Binary('{}'.format(y.b))
        y.c = Binary('{}'.format(y.c))
        y.d = Binary('{}'.format(y.d))
        y.e = Binary('{}'.format(y.e))
        y.f = Binary('{}'.format(y.f))
        y.g = Binary('{}'.format(y.g))
        y.h = Binary('{}'.format(y.h))
        y.i = Binary('{}'.format(y.i))
        y.j = Binary('{}'.format(y.j))

        if config.cluster == 2:
            cqm.add_constraint(y.a + y.b == 1)
            cqm.add_constraint(x.a + x.b == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b))
        if config.cluster == 3:
            cqm.add_constraint(y.a + y.b + y.c == 1)
            cqm.add_constraint(x.a + x.b + x.c == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c))
        if config.cluster == 4:
            cqm.add_constraint(y.a + y.b + y.c + y.d == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d))
        if config.cluster == 5:
            cqm.add_constraint(y.a + y.b + y.c + y.d + y.e == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d + x.e == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d + x.e * y.e))
        if config.cluster == 6:
            cqm.add_constraint(y.a + y.b + y.c + y.d + y.e + y.f == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d + x.e + x.f == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d + x.e * y.e + x.f * y.f))
        if config.cluster == 7:
            cqm.add_constraint(y.a + y.b + y.c + y.d + y.e + y.f + y.g == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d + x.e + x.f + x.g == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d + x.e * y.e + x.f * y.f + x.g * y.g))
        if config.cluster == 8:
            cqm.add_constraint(y.a + y.b + y.c + y.d + y.e + y.f + y.g + y.h == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d + x.e + x.f + x.g + x.h == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d + x.e * y.e + x.f * y.f + x.g * y.g + x.h * y.h))
        if config.cluster == 9:
            cqm.add_constraint(y.a + y.b + y.c + y.d + y.e + y.f + y.g + y.h + y.i == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d + x.e + x.f + x.g + x.h + x.i == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d + x.e * y.e + x.f * y.f + x.g * y.g + x.h * y.h + x.i * y.i))
        if config.cluster == 10:
            cqm.add_constraint(y.a + y.b + y.c + y.d + y.e + y.f + y.g + y.h + y.i + y.j == 1)
            cqm.add_constraint(x.a + x.b + x.c + x.d + x.e + x.f + x.g + x.h + x.i + x.j == 1)

            cqm.set_objective(-val * (x.a * y.a + x.b * y.b + x.c * y.c + x.d * y.d + x.e * y.e + x.f * y.f + x.g * y.g + x.h * y.h + x.i * y.i + x.j * y.j))

    solve_result = LeapHybridCQMSampler().sample_cqm(cqm, time_limit=5)
    print(solve_result)

    result = None
    for sampe, saf in solve_result.data(['sample', 'is_feasible']):
        if not saf:
            continue
        result = sampe

    # 返回退火结果最优解
    return result
def get_cluster(res, config):
    if config.cluster == 2:
        result_a = []
        result_b = []
        for (key, value) in res.items():
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if value != 1:
                continue
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
        return result_a, result_b
    if config.cluster == 3:
        result_a = []
        result_b = []
        result_c = []
        for (key, value) in res.items():
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if value != 1:
                continue
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
        return result_a, result_b, result_c
    if config.cluster == 4:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        for (key, value) in res.items():
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if value != 1:
                continue
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
        return result_a, result_b, result_c, result_d
    if config.cluster == 5:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        result_e = []
        print(res)
        for (key, value) in res.items():
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if value != 1:
                continue
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
            if area_cluster == 'e':
                result_e.append(area)
        return result_a, result_b, result_c, result_d, result_e
    if config.cluster == 6:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        result_e = []
        result_f = []
        for (key, value) in res.items():
            if value != 1:
                continue
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
            if area_cluster == 'e':
                result_e.append(area)
            if area_cluster == 'f':
                result_f.append(area)
        return result_a, result_b, result_c, result_d, result_e, result_f
    if config.cluster == 7:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        result_e = []
        result_f = []
        result_g = []
        for (key, value) in res.items():
            if value != 1:
                continue
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
            if area_cluster == 'e':
                result_e.append(area)
            if area_cluster == 'f':
                result_f.append(area)
            if area_cluster == 'g':
                result_g.append(area)
        return result_a, result_b, result_c, result_d, result_e, result_f, result_g

    if config.cluster == 8:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        result_e = []
        result_f = []
        result_g = []
        result_h = []
        for (key, value) in res.items():
            if value != 1:
                continue
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
            if area_cluster == 'e':
                result_e.append(area)
            if area_cluster == 'f':
                result_f.append(area)
            if area_cluster == 'g':
                result_g.append(area)
            if area_cluster == 'h':
                result_h.append(area)
        return result_a, result_b, result_c, result_d, result_e, result_f, result_g, result_h
    if config.cluster == 9:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        result_e = []
        result_f = []
        result_g = []
        result_h = []
        result_i = []
        for (key, value) in res.items():
            if value != 1:
                continue
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
            if area_cluster == 'e':
                result_e.append(area)
            if area_cluster == 'f':
                result_f.append(area)
            if area_cluster == 'g':
                result_g.append(area)
            if area_cluster == 'h':
                result_h.append(area)
            if area_cluster == 'i':
                result_i.append(area)
        return result_a, result_b, result_c, result_d, result_e, result_f, result_g, result_h, result_i
    if config.cluster == 10:
        result_a = []
        result_b = []
        result_c = []
        result_d = []
        result_e = []
        result_f = []
        result_g = []
        result_h = []
        result_i = []
        result_j = []
        for (key, value) in res.items():
            if value != 1:
                continue
            area = key.split("_")[0]
            area_cluster = key.split("_")[1]
            if area_cluster == 'a':
                result_a.append(area)
            if area_cluster == 'b':
                result_b.append(area)
            if area_cluster == 'c':
                result_c.append(area)
            if area_cluster == 'd':
                result_d.append(area)
            if area_cluster == 'e':
                result_e.append(area)
            if area_cluster == 'f':
                result_f.append(area)
            if area_cluster == 'g':
                result_g.append(area)
            if area_cluster == 'h':
                result_h.append(area)
            if area_cluster == 'i':
                result_i.append(area)
            if area_cluster == 'j':
                result_j.append(area)
        return result_a, result_b, result_c, result_d, result_e, result_f, result_g, result_h, result_i, result_j

def save_cluster(config):
    adj = pd.read_csv(config.adj_path, header=None)
    adj = adj.values

    feat = pd.read_csv(config.feet_path)
    feat = feat.values

    dict = {}
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0:
                dict[(i, j)] = adj[i, j]
    print(dict)
    res = QUBO_cluster(dict, config)
    cluster = get_cluster(res, config)

    for idx, item in enumerate(cluster):
        row_indices = [int(idx) for idx in item]
        selected_matrix = adj[row_indices][:, row_indices]
        df = pd.DataFrame(selected_matrix)
        adj_out_file = "data/cluster_{}/adj".format(config.cluster)
        if not os.path.exists(adj_out_file):
            os.makedirs(adj_out_file)
        df.to_csv(adj_out_file + "/adj_{}.csv".format(chr(ord('a') + idx)), index=False, header=False)

    for idx, item in enumerate(cluster):
        row_indices = [int(idx) for idx in item]
        selected_matrix = [[row[i] for i in row_indices] for row in feat]
        df = pd.DataFrame(selected_matrix)
        feat_out_file = "data/cluster_{}/feat".format(config.cluster)
        if not os.path.exists(feat_out_file):
            os.makedirs(feat_out_file)
        df.to_csv(feat_out_file + "/feat_{}.csv".format(chr(ord('a') + idx)), index=False, header=False)

if __name__ == '__main__':
   adj_out_file = "../data/cluster_{}/adj".format(2)
   df = pd.DataFrame()
   df.to_csv(adj_out_file + "/adj_{}.csv".format(chr(ord('a') + 0)), index=False, header=False)