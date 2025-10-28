import numpy as np
import time
import faiss

class KNNInnerProduct:
    """
    一个使用 NumPy 矩阵运算实现的 K-Nearest Neighbors 分类器。
    
    该版本使用 *内积 (dot product)* 作为相似度度量。
    K 个邻居被选为具有 *最高* 内积的训练样本。
    """
    
    def __init__(self, k=3):
        """
        初始化分类器。
        
        参数:
        k (int): 邻居的数量。
        """
        self.k = k
        # 确保 k 是正整数
        assert k >= 1, "k 必须是正整数"

    def fit(self, X_train, y_train):
        """
        "训练"模型，存储训练数据。
        
        参数:
        X_train (np.ndarray): 训练数据特征，形状 (n_train, n_features)
        y_train (np.ndarray): 训练数据标签，形状 (n_train,)
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        对新数据 X_test 进行预测。
        
        参数:
        X_test (np.ndarray): 测试数据特征，形状 (n_test, n_features)
        
        返回:
        np.ndarray: 预测的标签，形状 (n_test,)
        """
        
        # 1. 计算所有测试点和所有训练点之间的内积（相似度）
        # 这就是你想要的 "matrix product"
        # 形状: (n_test, n_train)
        similarity_matrix = self._compute_vectorized_similarity(X_test)
        
        # 2. 对每个测试点，找到 k 个最“近”邻的索引
        # "近" 在这里意味着 *最大* 的内积。
        # np.argsort 默认升序排序，所以我们对 *负* 的相似度矩阵排序，
        # 这样最小的负值（即最大的原始值）会排在最前面。
        # 形状: (n_test, k)
        knn_indices = np.argsort(-similarity_matrix, axis=1)[:, :self.k]
        
        # 3. 获取这 k 个邻居的标签
        # 形状: (n_test, k)
        knn_labels = self.y_train[knn_indices]
        
        # 4. 对每个测试点进行投票 (与 L2 版本相同)
        # 假设标签是 0, 1, 2, ...
        predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 
            axis=1, 
            arr=knn_labels
        )
        
        return predictions

    def _compute_vectorized_similarity(self, X_test):
        """
        使用矩阵乘法计算测试集和训练集之间的内积矩阵。
        """
        # (n_test, n_features) @ (n_features, n_train) -> (n_test, n_train)
        return np.dot(X_test, self.X_train.T)



class FaissKNNInnerProduct:
    
    def __init__(self, k=3):
        self.k = k
        self.index = None
    
    def fit(self, X_train, y_train):
        # 1. 确保数据是 float32 (Faiss 推荐)
        X_train_32 = X_train.astype(np.float32)
        
        # 2. (可选) Faiss 默认使用 L2，要使用内积，我们必须归一化
        #    或者使用 IndexFlatIP (IP = Inner Product)
        d = X_train_32.shape[1]
        self.index = faiss.IndexFlatIP(d) # IP = Inner Product
        
        # 3. (可选) 如果数据归一化了，IP 等价于余弦相似度
        # faiss.normalize_L2(X_train_32) # 如果需要余弦相似度
        
        # 4. 将训练数据添加到索引
        print("Faiss: 正在向索引添加数据...")
        self.index.add(X_train_32)
        print(f"Faiss: 添加了 {self.index.ntotal} 个向量")
        
        self.y_train = y_train # 仍然需要标签

    def predict(self, X_test):
        X_test_32 = X_test.astype(np.float32)
        # (可选) faiss.normalize_L2(X_test_32) # 如果需要余弦相似度
        
        # 1. 搜索!
        # D = 距离/相似度矩阵 (n_test, k)
        # I = 索引矩阵 (n_test, k)
        print("Faiss: 正在搜索...")
        start_search = time.time()
        
        # Faiss 的 IndexFlatIP 会返回 *最大* 内积
        D, I = self.index.search(X_test_32, self.k)
        
        end_search = time.time()
        print(f"Faiss: 搜索完毕，耗时: {end_search - start_search:.4f} 秒")
        
        # 2. 获取标签
        knn_labels = self.y_train[I] # I 的形状是 (n_test, k)
        
        # 3. 投票 (这部分和之前一样快)
        predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 
            axis=1, 
            arr=knn_labels
        )
        return predictions