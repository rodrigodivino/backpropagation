import {NN} from "./src/nn/nn.js";
import {ReLU} from "./src/activation-function/relu.js";
import {Linear} from "./src/activation-function/linear.js";

const nn = new NN(2, 3, new ReLU(), 2, new Linear(), 0.01);

nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [0, 0], [0, 0], [1, 1]]);
