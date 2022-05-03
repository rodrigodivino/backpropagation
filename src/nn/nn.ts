import {ActivationFunction} from "../activation-function/activation-function";
import {Linear} from "../activation-function/linear";

export class NN {
  private readonly outputActivationFunction: ActivationFunction = new Linear();
  
  private weights1: number[][];
  private weights2: number[][];
  
  constructor(
      private hiddenLayer: number,
      private hiddenActivationFunction: ActivationFunction
  ) {
    this.weights1 = new Array(5 + 1).fill(0).map(() => new Array(this.hiddenLayer).fill(0));
    this.weights2 = new Array(this.hiddenLayer).fill(0).map(() => new Array(1).fill(0));
  
    console.log("this.weights1", this.weights1);
    console.log("this.weights2", this.weights2);
  }
  
  
  train(inputs: number[][], outputs: number[]): void {
    const hiddenInducedLocalFields = this.multiPlaceholder([1, ...inputs], this.weights1);
    const activations = this.activateMatrixPlaceholder(hiddenInducedLocalFields, this.hiddenActivationFunction);
    const outputInducedLocalField = this.multiPlaceholder([1, ...activations], this.weights2);
    const output = this.activateMatrixPlaceholder(hiddenInducedLocalFields, this.outputActivationFunction);
  }
  
  multiPlaceholder(matrixLeft, matrixRight): any {
  
  }
  
  activateMatrixPlaceholder(matrix, activationFunction: ActivationFunction): any {
  
  }
}
