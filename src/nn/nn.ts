import {ActivationFunction} from "../activation-function/activation-function.js";
import {getMatrixMultiplication} from "../hooks/get-matrix-multiplication.js";
import {Linear} from "../activation-function/linear.js";
import {getAppliedMatrix} from "../hooks/get-applied-matrix.js";
import {getErrors} from "../hooks/get-errors.js";
import {getTransposedMatrix} from "../hooks/get-transposed-matrix.js";


export class NN {
  private readonly outputActivationFunction: ActivationFunction = new Linear();
  private readonly inputs = 2;
  private readonly outputs = 2;
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
    // (N x (2+1))
    const inputSetPlusBias = inputSet.map(inputs => [1, ...inputs]);
    console.log("inputSet", inputSet);
    console.log("inputSetPlusBias", inputSetPlusBias);
    
    // (N x (2+1)) * ((2+1) x 3) = (N x 3)
    const hiddenLayerInducedLocalFieldsSet = getMatrixMultiplication(inputSetPlusBias, this.weights1);
    
    // (N x 3)
    const hiddenLayerActivationsSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenActivationFunction.activate
    );
    
    // (N x (3+1))
    const hiddenLayerActivationSetPlusBias = hiddenLayerActivationsSet.map(a => [1, ...a]);
    
    console.log("hiddenLayerInducedLocalFieldsSet", hiddenLayerInducedLocalFieldsSet);
    console.log("hiddenLayerActivationsSet", hiddenLayerActivationsSet);
    console.log("hiddenLayerActivationSetPlusBias", hiddenLayerActivationSetPlusBias);
    
    // (N x (3+1)) * ((3+1) x 2) = (N x 2)
    const outputLayerInducedLocalFieldsSet = getMatrixMultiplication(
        hiddenLayerActivationSetPlusBias,
        this.weights2
    );
    
    // (N x 2)
    const outputLayerActivationsSet = getAppliedMatrix(
        outputLayerInducedLocalFieldsSet,
        this.outputActivationFunction.activate
    );
    
    console.log("outputLayerInducedLocalFieldsSet", outputLayerInducedLocalFieldsSet);
    console.log("outputLayerActivationsSet", outputLayerActivationsSet);
    
    console.log("expectedOutputSet", expectedOutputSet);
    
    // (N x 2)
    const errorsSet = getErrors(outputLayerActivationsSet, expectedOutputSet);
    
    console.log("errorsSet", errorsSet);
    // (N x 2)
    const outputLayerLocalGradientsSet = this.calculateOutputLocalGradient(errorsSet, outputLayerInducedLocalFieldsSet);
    
    console.log("outputLayerLocalGradientsSet", outputLayerLocalGradientsSet);
    
    console.log('--- obtaining weights by multiplying ---');
    console.log('the transposed of hiddenLayerActivationSetPlusBias', hiddenLayerActivationSetPlusBias);
    console.log('the outputLayerLocalGradientsSet', outputLayerLocalGradientsSet);
    // t((3+1) x N)t * (N x 2) = ((3+1) x 2)
    const outputLayerWeightAdjustmentMatrix = getMatrixMultiplication(
        getTransposedMatrix(hiddenLayerActivationSetPlusBias),
        outputLayerLocalGradientsSet
    );
    
    console.log("outputLayerWeightAdjustmentMatrix", outputLayerWeightAdjustmentMatrix);
    // ((3+1) x 2)
    const averageOutputLayerWeightAdjustmentMatrix = getAppliedMatrix(
        outputLayerWeightAdjustmentMatrix,
        d => d / inputSet.length
    );
    
    console.log("averageOutputLayerWeightAdjustmentMatrix", averageOutputLayerWeightAdjustmentMatrix);
  
    // (N x 3)
    const hiddenLayerLocalGradientsSet = this.calculateHiddenLayerLocalGradientSet(
        outputLayerLocalGradientsSet,
        hiddenLayerInducedLocalFieldsSet
    );
  
    console.log("hiddenLayerLocalGradientsSet", hiddenLayerLocalGradientsSet);
    
    // ((2+1) x 3)
    const hiddenLayerWeightAdjustmentMatrix = getMatrixMultiplication(
        // t(N x (2+1))t
        getTransposedMatrix(inputSetPlusBias),
        // (N x 3)
        hiddenLayerLocalGradientsSet
    );
  
    console.log("hiddenLayerWeightAdjustmentMatrix", hiddenLayerWeightAdjustmentMatrix);
    
    const averageHiddenLayerWeightAdjustmentMatrix = getAppliedMatrix(
        hiddenLayerWeightAdjustmentMatrix,
        d => d / inputSet.length
    );
  
    console.log("averageHiddenLayerWeightAdjustmentMatrix", averageHiddenLayerWeightAdjustmentMatrix);
  
  }
  
  // (N x 2)
  calculateOutputLocalGradient(
      //(N x 2)
      errorsSet: number[][],
      
      // (N x 2)
      outputLocalInducedFieldsSet: number[][]
  ): any {
    const outputLayerDerivativesSet = getAppliedMatrix(
        outputLocalInducedFieldsSet,
        this.outputActivationFunction.derivative
    );
    
    // (N x 2)
    return errorsSet.map((errors, n) => {
      const derivatives = outputLayerDerivativesSet[n];
      return errors.map((errorOfNeuron, i) => {
        const derivativeOfNeuron = derivatives[i];
        return errorOfNeuron * derivativeOfNeuron;
      });
    });
  }
  
  // (N x 3)
  private calculateHiddenLayerLocalGradientSet(
      // (N x 2)
      outputLayerLocalGradientsSet: any,
      // (N x 3)
      hiddenLayerInducedLocalFieldsSet: number[][]
  ): any {
    // (N x 3)
    const hiddenLayerDerivativesSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenActivationFunction.derivative
    );
    
    // Remove bias from computation because it is not connected to layer 1, so it doesn't need a local gradient
    // (3 x 2)
    const weights2WithoutBias = this.weights2.slice(1)
    
    // (N x 3)
    const backpropagatedGradientsSet = getMatrixMultiplication(
        // (N x 2)
        outputLayerLocalGradientsSet,
        
        // t(2 x 3)t
        getTransposedMatrix(weights2WithoutBias)
    )
  
    // (N x 3)
    return backpropagatedGradientsSet.map((backpropagatedGradients, n) => {
      const derivatives = hiddenLayerDerivativesSet[n];
      return backpropagatedGradients.map((backPropagatedGradientOfNeuron, i) => {
        const derivativeOfNeuron = derivatives[i];
        return backPropagatedGradientOfNeuron * derivativeOfNeuron;
      });
    });
  }
}
