import {ActivationFunction} from "../activation-function/activation-function";

export class NN {
  private weights1: number[][];
  private weights2: number[][];
  
  constructor(
      private hiddenLayer: number,
      private activationFunction: ActivationFunction
  ) {
    this.weights1 = new Array(5).fill(0).map(() => new Array(this.hiddenLayer).fill(0));
    this.weights2 = new Array(this.hiddenLayer).fill(0).map(() => new Array(1).fill(0));
  
    console.log("this.weights1", this.weights1);
    console.log("this.weights2", this.weights2);
  }
}
