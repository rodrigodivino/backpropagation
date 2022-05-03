import {NN} from "./src/nn/nn.js";
import {getMatrixMultiplication} from "./src/hooks/get-matrix-multiplication.js";
import {ReLU} from "./src/activation-function/relu.js";

const nn = new NN(100, new ReLU());
