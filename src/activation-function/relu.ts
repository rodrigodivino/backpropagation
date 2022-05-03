import {ActivationFunction} from "./activation-function";

export class ReLU implements ActivationFunction {
  activate(input: number): number {
    if(input <= 0) return 0;
    else return input
  }
  
  derivative(input: number): number {
    if(input <= 0) return 0;
    else return 1
  }
}
