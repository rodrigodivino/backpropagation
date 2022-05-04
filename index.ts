import {NN} from "./src/nn/nn.js";
import {ReLU} from "./src/activation-function/relu.js";
import {Linear} from "./src/activation-function/linear.js";

const nn = new NN(1, 3, new ReLU(), 1, new Linear(), 0.01);

for(let i = 0; i < 100; i ++){
  console.log("---i---", i);
  nn.train(new Array(100).fill(0).map((_,i) =>[i]), new Array(100).fill(0).map((_,i) =>[i % 2]));
}
