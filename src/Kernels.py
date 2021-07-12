from grakel.kernels import WeisfeilerLehman, VertexHistogram


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
