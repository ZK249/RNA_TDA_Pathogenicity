import RNA

class RNAStructureModeler:
    def __init__(self, temperature=37.0):
        self.md = RNA.md()
        self.md.temperature = temperature
    
    def get_bpp_matrix(self, sequence: str):
        """
        返回碱基配对概率矩阵 (n x n)
        fc.bpp()[i][j] = 碱基i与j配对的概率
        """
        fc = RNA.fold_compound(sequence, self.md)
        fc.pf()  # 配分函数计算，必须调用才能获取概率
        bpp = fc.bpp()
        # 转换为完整numpy矩阵
        import numpy as np
        n = len(sequence)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    mat[i][j] = bpp[i+1][j+1]  # ViennaRNA是1-indexed
                    mat[j][i] = mat[i][j]
        return mat
    
    def get_mfe_structure(self, sequence: str):
        """返回MFE结构和能量"""
        fc = RNA.fold_compound(sequence, self.md)
        mfe_structure, mfe_energy = fc.mfe()
        return mfe_structure, mfe_energy