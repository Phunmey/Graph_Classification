from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, ShortestPath


class Kernel:
    def __init__(self, kernel, name):
        self.kernel = kernel
        self.name = name

    def get_kernel(self):
        return self.kernel

    def get_name(self):
        return self.name

    def get_k_test(self, g_test):
        return self.kernel.transform(g_test)

    def get_k_train(self, g_train):
        return self.kernel.fit_transform(g_train)


class WLehman(Kernel):
    __kernel = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    __name = "Weisfeiler Lehman"

    def __init__(self):
        super().__init__(self.__kernel, self.__name)


class WLehmanOptimal(Kernel):
    __kernel = WeisfeilerLehmanOptimalAssignment(n_iter=4, normalize=True)
    __name = "Weisfeiler Lehman Optimal Assignment"

    def __init__(self):
        super().__init__(self.__kernel, self.__name)


class ShortPath(Kernel):
    __kernel = ShortestPath(normalize=True)
    __name = "Shortest Path"

    def __init__(self):
        super().__init__(self.__kernel, self.__name)
