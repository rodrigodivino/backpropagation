import { NN } from "./src/nn/nn.js";
import { ReLU } from "./src/activation-function/relu.js";
var nn = new NN(5, new ReLU());
nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]);
//# sourceMappingURL=index.js.map