import {ActivationFunction} from "./activation-function.js";

export class Linear implements ActivationFunction {
  activate(input: number): number {
    return input
  }
  
  derivative(input: number): number {
    return 1;
  }
}
