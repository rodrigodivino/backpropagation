import {NN} from "./src/nn/nn.js";
import {getMatrixMultiplication} from "./src/hooks/get-matrix-multiplication.js";
import {ReLU} from "./src/activation-function/relu.js";

const nn = new NN(5, new ReLU());

nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
