import {NN} from "./src/nn/nn.js";
import {Linear} from "./src/activation-function/linear.js";
import {data} from './data/mamografia.js'

declare var d3: any;

const nn = new NN(5, 10, new Linear(), 1, new Linear(), 0.1);

console.log("data", data);

const maxBirads = d3.max(data, (d) => d.birads);
const maxIdade = d3.max(data, (d) => d.idade);
const maxForma = d3.max(data, (d) => d.forma);
const maxMargem = d3.max(data, (d) => d.margem);
const maxDensidade = d3.max(data, (d) => d.densidade);

console.log("maxBirads", maxBirads);

const inputs = data.map(d => {
  return [d.birads / maxBirads, d.idade / maxIdade, d.forma / maxForma, d.margem / maxMargem, d.densidade / maxDensidade]
})

const outputs = data.map(d => {
  return [d.classe]
})


console.log("inputs", inputs);

console.log("outputs", outputs);

for(let i = 0; i < 100; i ++){
  console.log("---i---", i);
  nn.train(inputs, outputs);
}
