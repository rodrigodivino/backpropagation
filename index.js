import { NN } from "./src/nn/nn.js";
import { Linear } from "./src/activation-function/linear.js";
var nn = new NN(2, 100, new Linear(), 1, new Linear(), 0.01);
for (var i = 0; i < 1000; i++) {
    console.log("---i---", i);
    nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]);
}
//# sourceMappingURL=index.js.map