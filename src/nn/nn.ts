import {ActivationFunction} from "../activation-function/activation-function";
import {Linear} from "../activation-function/linear";

export class NN {
  private readonly outputActivationFunction: ActivationFunction = new Linear();
  private readonly learningRate = 0.01;
  
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
  
  
  train(inputs: number[][], expectedOutput: number[]): void {
    const hiddenLayerInducedLocalFields = this.multiPlaceholder([1, ...inputs], this.weights1);
    const hiddenLayerActivations = this.activateMatrixPlaceholder(
        hiddenLayerInducedLocalFields,
        this.hiddenActivationFunction
    );
    const outputInducedLocalField = this.multiPlaceholder([1, ...hiddenLayerActivations], this.weights2);
    const output = this.activateMatrixPlaceholder(hiddenLayerInducedLocalFields, this.outputActivationFunction);
    
    const errors = this.calculateErrorsPlaceholder(output, expectedOutput);
    const meanErrors = this.calculateMeanErrorsPlaceholder(errors);
    
    const outputLayerLocalGradients = this.calculateOutputLocalGradient(meanErrors);
    const outputLayerWeightAdjustmentMatrix = this.calculateOutputWeightAdjustmentMatrix(
        hiddenLayerActivations,
        outputLayerLocalGradients
    );
    
    const hiddenLayerWeightAdjustmentMatrix = this.backpropagateOutputLocalGradients(
        hiddenLayerActivations,
        outputLayerLocalGradients
    );
  }
  
  multiPlaceholder(matrixLeft, matrixRight): any {
  
  }
  
  activateMatrixPlaceholder(matrix, activationFunction: ActivationFunction): any {
  
  }
  
  calculateErrorsPlaceholder(output, expectedOutput): number[][] {
    return [[0]];
  }
  
  calculateMeanErrorsPlaceholder(error: number[][]): number[] {
    return [0];
  }
  
  calculateOutputLocalGradient(meanErrors: number[]): any {
  
  }
  
  calculateOutputWeightAdjustmentMatrix(hiddenLayerActivations: number[][], outputLocalGradients: number[][]): any {
  
  }
  
  backpropagateOutputLocalGradients(...args: any): any {
  
  }
}
