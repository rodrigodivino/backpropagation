import { NN } from "./src/nn/nn.js";
import { Linear } from "./src/activation-function/linear.js";
import { data } from './data/mamografia.js';
var nn = new NN(5, 10, new Linear(), 1, new Linear(), 0.1);
console.log("data", data);
var maxBirads = d3.max(data, function (d) { return d.birads; });
var maxIdade = d3.max(data, function (d) { return d.idade; });
var maxForma = d3.max(data, function (d) { return d.forma; });
var maxMargem = d3.max(data, function (d) { return d.margem; });
var maxDensidade = d3.max(data, function (d) { return d.densidade; });
console.log("maxBirads", maxBirads);
var inputs = data.map(function (d) {
    return [d.birads / maxBirads, d.idade / maxIdade, d.forma / maxForma, d.margem / maxMargem, d.densidade / maxDensidade];
});
var outputs = data.map(function (d) {
    return [d.classe];
});
console.log("inputs", inputs);
console.log("outputs", outputs);
for (var i = 0; i < 100; i++) {
    console.log("---i---", i);
    nn.train(inputs, outputs);
}
//# sourceMappingURL=index.js.map