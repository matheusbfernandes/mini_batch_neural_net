import numpy as np
import math
import pickle
import gzip
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class MLP(object):
    def __init__(self, dados_x, dados_y, dimensao_layers, epoch=100, minibatch_tamanho=32, taxa_aprendizado=0.1, camadas_escondidas="sigmoid", mostrar_grafico_custo=False):
        self.X = dados_x
        self.Y = dados_y
        self.epoch = epoch
        self.minibatch_tamanho = minibatch_tamanho
        self.taxa_aprendizado = taxa_aprendizado
        self.dimensao_layers = dimensao_layers
        self.camadas_escondidas = camadas_escondidas
        self.mostrar_grafico_custo = mostrar_grafico_custo
        self.layers = len(dimensao_layers) - 1
        self.parametros = self._inicializar_parametros

    @staticmethod
    def _softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    @staticmethod
    def _tanh(z):
        return (2 / (1 + np.exp(-2 * z))) - 1

    @staticmethod
    def _tanh_backward(a):
        return 1 - a ** 2

    @staticmethod
    def _relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def _relu_backward(z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_backward(a):
        return a * (1 - a)

    @property
    def _inicializar_parametros(self):
        parametros = {}
        for l in range(1, len(self.dimensao_layers)):
            parametros["W" + str(l)] = np.random.randn(self.dimensao_layers[l], self.dimensao_layers[l - 1]) * math.sqrt(2 / self.dimensao_layers[l - 1])
            parametros["b" + str(l)] = np.zeros((self.dimensao_layers[l], 1))

        return parametros

    @staticmethod
    def computar_custo(y_mini_batch, y_chapeu):
        return -np.sum(y_mini_batch * np.log(y_chapeu)) / y_mini_batch.shape[1]

    def _forward_propagation(self, dados_entrada=None):
        if dados_entrada is None:
            dados_entrada = self.X
        z1 = np.dot(self.parametros["W1"], dados_entrada) + self.parametros["b1"]
        resultado_forward = {
            "A0": dados_entrada,
            "Z1": z1,
        }
        if self.camadas_escondidas == "relu":
            resultado_forward["A1"] = self._relu(z1)
        elif self.camadas_escondidas == "tanh":
            resultado_forward["A1"] = self._tanh(z1)
        else:
            resultado_forward["A1"] = self._sigmoid(z1)
        for l in range(2, self.layers):
            resultado_forward["Z" + str(l)] = np.dot(self.parametros["W" + str(l)], resultado_forward["A" + str(l - 1)]) + self.parametros["b" + str(l)]
            if self.camadas_escondidas == "relu":
                resultado_forward["A" + str(l)] = self._relu(resultado_forward["Z" + str(l)])
            elif self.camadas_escondidas == "tanh":
                resultado_forward["A" + str(l)] = self._tanh(resultado_forward["Z" + str(l)])
            else:
                resultado_forward["A" + str(l)] = self._sigmoid(resultado_forward["Z" + str(l)])
        resultado_forward["Z" + str(self.layers)] = np.dot(self.parametros["W" + str(self.layers)], resultado_forward["A" + str(self.layers - 1)]) + self.parametros["b" + str(self.layers)]
        resultado_forward["A" + str(self.layers)] = self._softmax(resultado_forward["Z" + str(self.layers)])

        return resultado_forward

    def _backward_propagation(self, resultado_forward, mini_batch):
        num_exemplos = mini_batch.shape[1]
        dz = resultado_forward["A" + str(self.layers)] - mini_batch
        grads = {
            "dW" + str(self.layers): np.dot(dz, resultado_forward["A" + str(self.layers - 1)].T) / num_exemplos,
            "db" + str(self.layers): np.sum(dz, axis=1, keepdims=True) / num_exemplos
        }
        for i in reversed(range(1, self.layers)):
            if self.camadas_escondidas == "relu":
                dz = np.dot(self.parametros["W" + str(i + 1)].T, dz) * self._relu_backward(resultado_forward["Z" + str(i)])
            elif self.camadas_escondidas == "tanh":
                dz = np.dot(self.parametros["W" + str(i + 1)].T, dz) * self._tanh_backward(resultado_forward["A" + str(i)])
            else:
                dz = np.dot(self.parametros["W" + str(i + 1)].T, dz) * self._sigmoid_backward(resultado_forward["A" + str(i)])
            grads["dW" + str(i)] = np.dot(dz, resultado_forward["A" + str(i - 1)].T) / num_exemplos
            grads["db" + str(i)] = np.sum(dz, axis=1, keepdims=True) / num_exemplos

        return grads

    def predizer(self, dados_entrada):
        resultado_forward = self._forward_propagation(dados_entrada)

        return (resultado_forward["A" + str(self.layers)] == np.amax(resultado_forward["A" + str(self.layers)], axis=0)).astype(float)

    def taxa_acertos(self, dados_entrada, dados_saida):
        num_dados = dados_entrada.shape[1]
        p = self.predizer(dados_entrada)
        total_positivos = np.sum(np.all((p == dados_saida), axis=0)).astype(float)

        return total_positivos / num_dados

    def _obter_mini_batches(self):
        num_exemplos = self.X.shape[1]
        permutacao = list(np.random.permutation(num_exemplos))
        x_aleatorio = self.X[:, permutacao]
        y_aleatorio = self.Y[:, permutacao].reshape((self.Y.shape[0], num_exemplos))

        mini_batches = []
        mini_batches_completos = math.floor(num_exemplos / self.minibatch_tamanho)
        for k in range(mini_batches_completos):
            mini_batch_x = x_aleatorio[:, k * self.minibatch_tamanho:(k + 1) * self.minibatch_tamanho]
            mini_batch_y = y_aleatorio[:, k * self.minibatch_tamanho:(k + 1) * self.minibatch_tamanho]
            mini_batches.append((mini_batch_x, mini_batch_y))

        if (num_exemplos % self.minibatch_tamanho) != 0:
            mini_batch_x = x_aleatorio[:, self.minibatch_tamanho * mini_batches_completos:num_exemplos]
            mini_batch_y = y_aleatorio[:, self.minibatch_tamanho * mini_batches_completos:num_exemplos]
            mini_batches.append((mini_batch_x, mini_batch_y))

        return mini_batches

    def treinar(self):
        mini_batches = self._obter_mini_batches()
        custos = []
        for i in range(self.epoch):
            custo = 0
            for mini_batch in mini_batches:
                resultado_forward = self._forward_propagation(mini_batch[0])
                custo = self.computar_custo(mini_batch[1], resultado_forward["A" + str(self.layers)])
                resultado_backward = self._backward_propagation(resultado_forward, mini_batch[1])
                for l in range(1, self.layers + 1):
                    self.parametros["W" + str(l)] -= self.taxa_aprendizado * resultado_backward["dW" + str(l)]
                    self.parametros["b" + str(l)] -= self.taxa_aprendizado * resultado_backward["db" + str(l)]
            if self.mostrar_grafico_custo and (i % 5 == 0):
                print("Custo: {:.3f}, epoch: {:d}".format(custo, i))
                custos.append(custo)

        if self.mostrar_grafico_custo:
            plt.plot(np.squeeze(custos))
            plt.ylabel("custo")
            plt.xlabel("epochs")
            plt.title("Taxa de aprendizado=" + str(self.taxa_aprendizado))
            plt.show()


def main():
    with gzip.open("mnist.pkl.gz", "rb") as f:
        if sys.version_info.major > 2:
            dados_treino, dados_validacao, dados_teste = pickle.load(f, encoding='latin1')
        else:
            dados_treino, dados_validacao, dados_teste = pickle.load(f)
    treino_x, treino_y = dados_treino
    plt.imshow(treino_x[0].reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
    treino_x = treino_x.T
    treino_y_one_hot = np.eye(10)[treino_y]
    treino_y_one_hot = treino_y_one_hot.T
    mlp = MLP(treino_x, treino_y_one_hot, [784, 18, 10], 40, 64, 0.06, "relu", True)
    mlp.treinar()
    print(mlp.taxa_acertos(treino_x, treino_y_one_hot))


if __name__ == "__main__":
    main()
