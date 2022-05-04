import {ActivationFunction} from "./activation-function.js";

export class Sigmoid implements ActivationFunction {
  activate(input: number): number {
    return 1 / (1 + Math.E**(-input));
  }
  
  derivative(input: number): number {
    const activation = 1 / (1 + Math.E**(-input));
    return activation * (1- activation);
  }
  
}
