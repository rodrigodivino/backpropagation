import {ActivationFunction} from "../activation-function/activation-function.js";
import {getMatrixMultiplication} from "../hooks/get-matrix-multiplication.js";
import {Linear} from "../activation-function/linear.js";
import {getAppliedMatrix} from "../hooks/get-applied-matrix.js";


export class NN {
  private readonly outputActivationFunction: ActivationFunction = new Linear();
  private readonly inputs = 2;
  private readonly outputs = 1;
  private readonly learningRate = 0.01;
  
  private weights1: number[][];
  private weights2: number[][];
  
  constructor(
      private hiddenLayer: number,
      private hiddenActivationFunction: ActivationFunction
  ) {
    this.weights1 = new Array(this.inputs + 1).fill(0).map(() => new Array(this.hiddenLayer).fill(0).map(() => Math.random()));
    this.weights2 = new Array(this.hiddenLayer + 1).fill(0).map(() => new Array(this.outputs).fill(0).map(() => Math.random()));
    
    console.log("this.weights1", this.weights1);
    console.log("this.weights2", this.weights2);
  }
  
  
  train(inputs: number[][], expectedOutput: number[]): void {
    const hiddenLayerInducedLocalFields = getMatrixMultiplication(inputs.map(i => [1, ...i]), this.weights1);
    const hiddenLayerActivations = getAppliedMatrix(
        hiddenLayerInducedLocalFields,
        this.hiddenActivationFunction.activate
    );
    
    const outputInducedLocalField = getMatrixMultiplication(hiddenLayerActivations.map(a => [1, ...a]), this.weights2);
    const output = getAppliedMatrix(hiddenLayerInducedLocalFields, this.outputActivationFunction.activate);
    
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
