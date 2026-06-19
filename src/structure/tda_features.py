import numpy as np
from ripser import ripser
from persim import PersistenceImager, persistent_entropy
from scipy.spatial.distance import squareform

class RNATopologicalFingerprint:
    def __init__(self, max_dim=1):
        self.max_dim = max_dim  # 计算0维和1维持久同调
    
    def bpp_to_distance_matrix(self, bpp_matrix: np.ndarray):
        """
        将碱基配对概率转换为距离矩阵：
        - 高概率配对 → 短距离（0.1）
        - 不配对 → 长距离（1.0）
        - 骨架连接（i, i+1）固定短距离
        """
        n = len(bpp_matrix)
        dist = np.ones((n, n))  # 默认不连接
        
        # 骨架连接：相邻核苷酸
        for i in range(n-1):
            dist[i, i+1] = 0.1
            dist[i+1, i] = 0.1
        
        # 碱基配对：概率越高，距离越近
        for i in range(n):
            for j in range(i+1, n):
                if bpp_matrix[i, j] > 0.01:  # 过滤噪声
                    dist[i, j] = 1.0 - bpp_matrix[i, j]
                    dist[j, i] = dist[i, j]
        return dist
    
    def extract_features(self, bpp_matrix: np.ndarray):
        """
        提取持久同调特征向量
        """
        dist_mat = self.bpp_to_distance_matrix(bpp_matrix)
        
        # Ripser计算持久同调，输入距离矩阵
        # distance_matrix=True 表示输入是距离矩阵而非点云
        diagrams = ripser(dist_mat, distance_matrix=True, maxdim=self.max_dim)['dgms']
        
        features = {}
        
        # 0维特征（连通分量）
        dgm0 = diagrams[0]
        if len(dgm0) > 0:
            # 排除无穷远点
            finite0 = dgm0[dgm0[:, 1] != np.inf]
            features['betti0_count'] = len(finite0)
            features['persistence_entropy_0'] = persistent_entropy(finite0)
            features['mean_persistence_0'] = np.mean(finite0[:, 1] - finite0[:, 0]) if len(finite0) > 0 else 0
        
        # 1维特征（环/茎环结构）
        dgm1 = diagrams[1]
        if len(dgm1) > 0:
            features['betti1_count'] = len(dgm1)
            features['persistence_entropy_1'] = persistent_entropy(dgm1)
            features['mean_persistence_1'] = np.mean(dgm1[:, 1] - dgm1[:, 0])
            features['max_persistence_1'] = np.max(dgm1[:, 1] - dgm1[:, 0])
        
        # 持久景观（Persistence Landscape）向量化
        # 用于输入神经网络
        features['landscape'] = self._landscape_vector(diagrams)
        
        return features
    
    def _landscape_vector(self, diagrams, resolution=20):
        """将持久图转化为固定长度向量"""
        # 简化版：用持久熵+Betti数曲线采样
        # 实际可用Persim的PersistenceImager
        from persim import PersistenceImager
        pimgr = PersistenceImager(pixel_size=0.1)
        # 这里仅作示意，返回拼接向量
        return np.zeros(resolution * 2)  # 占位