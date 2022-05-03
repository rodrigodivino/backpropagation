import {ActivationFunction} from "../activation-function/activation-function.js";
import {getMatrixMultiplication} from "../hooks/get-matrix-multiplication.js";
import {Linear} from "../activation-function/linear.js";
import {getAppliedMatrix} from "../hooks/get-applied-matrix.js";
import {getErrors} from "../hooks/get-errors.js";
import {getTransposedMatrix} from "../hooks/get-transposed-matrix.js";


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
    this.weights1 =
        new Array(this.inputs + 1).fill(0).map(() => new Array(this.hiddenLayer).fill(0).map(() => Math.random()));
    this.weights2 =
        new Array(this.hiddenLayer + 1).fill(0).map(() => new Array(this.outputs).fill(0).map(() => Math.random()));
    
    console.log("this.weights1", this.weights1);
    console.log("this.weights2", this.weights2);
  }
  
  
  train(inputSet: number[][], expectedOutputSet: number[][]): void {
    const inputSetPlusBias = inputSet.map(inputs => [1, ...inputs]);
    console.log("inputSet", inputSet);
    console.log("inputSetPlusBias", inputSetPlusBias);
    
    const hiddenLayerInducedLocalFieldsSet = getMatrixMultiplication(inputSetPlusBias, this.weights1);
    const hiddenLayerActivationsSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenActivationFunction.activate
    );
    
    const hiddenLayerActivationSetPlusBias = hiddenLayerActivationsSet.map(a => [1, ...a]);
    
    console.log("hiddenLayerInducedLocalFieldsSet", hiddenLayerInducedLocalFieldsSet);
    console.log("hiddenLayerActivationsSet", hiddenLayerActivationsSet);
    console.log("hiddenLayerActivationSetPlusBias", hiddenLayerActivationSetPlusBias);
    
    const outputLayerInducedLocalFieldsSet = getMatrixMultiplication(
        hiddenLayerActivationSetPlusBias,
        this.weights2
    );
    const outputLayerActivationsSet = getAppliedMatrix(
        outputLayerInducedLocalFieldsSet,
        this.outputActivationFunction.activate
    );
    
    console.log("outputLayerInducedLocalFieldsSet", outputLayerInducedLocalFieldsSet);
    console.log("outputLayerActivationsSet", outputLayerActivationsSet);
    
    console.log("expectedOutputSet", expectedOutputSet);
    
    const errorsSet = getErrors(outputLayerActivationsSet, expectedOutputSet);
    
    console.log("errorsSet", errorsSet);
    
    const outputLayerLocalGradientsSet = this.calculateOutputLocalGradient(errorsSet, outputLayerInducedLocalFieldsSet);
    
    console.log("outputLayerLocalGradientsSet", outputLayerLocalGradientsSet);
    
    console.log('--- obtaining weights by multiplying ---')
    console.log('the transposed of hiddenLayerActivationSetPlusBias', hiddenLayerActivationSetPlusBias)
    console.log('the outputLayerLocalGradientsSet', outputLayerLocalGradientsSet)
    
    const outputLayerWeightAdjustmentMatrix = getMatrixMultiplication(
        getTransposedMatrix(hiddenLayerActivationSetPlusBias),
        outputLayerLocalGradientsSet
    );
    
    console.log("outputLayerWeightAdjustmentMatrix", outputLayerWeightAdjustmentMatrix);
    
    const averageOutputLayerWeightAdjustmentMatrix = getAppliedMatrix(outputLayerWeightAdjustmentMatrix, d => d / inputSet.length)
  
    console.log("averageOutputLayerWeightAdjustmentMatrix", averageOutputLayerWeightAdjustmentMatrix);
    
    // const hiddenLayerLocalGradientSet = getMatrixMultiplication()
    //
    // const hiddenLayerWeightAdjustmentMatrix = this.backpropagateOutputLocalGradients(
    //     hiddenLayerActivationsSet,
    //     outputLayerLocalGradientsSet
    // );
  }
  
  calculateErrorsPlaceholder(output, expectedOutput): number[][] {
    return [[0]];
  }
  
  calculateMeanErrorsPlaceholder(error: number[][]): number[] {
    return [0];
  }
  
  calculateOutputLocalGradient(errorsSet: number[][], outputLocalInducedFieldsSet: number[][]): any {
    const outputDerivativesSet = getAppliedMatrix(
        outputLocalInducedFieldsSet,
        this.outputActivationFunction.derivative
    );
    return errorsSet.map((errors, n) => {
      const derivatives = outputDerivativesSet[n];
      return errors.map((errorOfNeuron, i) => {
        const derivativeOfNeuron = derivatives[i];
        return errorOfNeuron * derivativeOfNeuron;
      });
    });
  }
  
  calculateOutputWeightAdjustmentMatrix(hiddenLayerActivations: number[][], outputLocalGradients: number[][]): any {
  
  }
  
  backpropagateOutputLocalGradients(...args: any): any {
  
  }
}
