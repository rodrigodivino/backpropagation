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
    /**
     * @var this.weights1
     * @description The matrix of size [1+2,3] of (i,j) weights between the input layer and the hidden layer
     *    Each weight (i,j) connects the 1 bias and the 2 neurons (i) of the input layer
     *    to the 3 neurons (j) of the hidden layer
     */
    this.weights1 =
        new Array(this.inputs + 1).fill(0).map(() => new Array(this.hiddenLayer).fill(0).map(() => Math.random()));
  
    /**
     * @var this.weights2
     * @description The matrix of size [1+3,2] of (i,j) weights between the hidden layer and the output layer
     *    Each weight (i,j) connects the 1 bias and the 3 neurons (i) of the hidden layer
     *    to the 2 neurons (j) of the output layer
     */
    this.weights2 =
        new Array(this.hiddenLayer + 1).fill(0).map(() => new Array(this.outputs).fill(0).map(() => Math.random()));
  }
  
  /**
   * @description a weight update with the provided batch
   * @param inputSet - The batch to use for weight update
   * @param expectedOutputSet - The expected output of neurons for each entry in the batch
   */
  train(inputSet: number[][], expectedOutputSet: number[][]): void {
    /**
     * @var inputSetPlusBias
     * @description The input set [N, 2], pre-pended with a bias [N, 1+2]
     * Size [N, 1+2] (N entries, each with 1 fixed bias and 2 inputs)
     */
    const inputSetPlusBias = inputSet.map(inputs => [1, ...inputs]);
    
    /**
     * @var hiddenLayerInducedLocalFieldsSet
     * @description The set of induced local fields for the hidden layer
     * Multiplies inputSetPlusBias [N, 1+2] with this.weights1 [1+2, 3],
     * Size [N, 3] (N entries, each with 3 induced local fields, one for each neuron in the hidden layer)
     */
    const hiddenLayerInducedLocalFieldsSet = getMatrixMultiplication(inputSetPlusBias, this.weights1);
    
    /**
     * @var hiddenLayerActivationSet
     * @description The set of activations for the hidden layer
     * Applies the hidden layer activation function to the induced local fields of the hidden neurons
     * Size [N, 3] (N entries, each with 3 activations, one for each neuron in the hidden layer
     */
    const hiddenLayerActivationsSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenActivationFunction.activate
    );
    
    /**
     * @var hiddenLayerActivationSetPlusBias
     * @description The hiddenLayerActivationsSet [N, 3], pre-pended with a bias [N, 1+3]
     * Size [N, 1+3] (N entries, each with 1 fixed bias and 3 activations, one for each neuron in the hidden layer)
     */
    const hiddenLayerActivationSetPlusBias = hiddenLayerActivationsSet.map(a => [1, ...a]);
    
    
    /**
     * @var outputLayerInducedLocalFieldsSet
     * @description The set of induced local fields for the output layer
     * Multiplies hiddenLayerActivationSetPlusBias [N, 1+3] with this.weights2 [1+3, 2],
     * Size [N, 2] (N entries, each with 2 induced local fields, one for each neuron in the output layer)
     */
    const outputLayerInducedLocalFieldsSet = getMatrixMultiplication(
        hiddenLayerActivationSetPlusBias,
        this.weights2
    );
    
    /**
     * @var outputLayerActivationsSet
     * @description The set of activations for the output layer
     * Applies the output layer activation function to the induced local fields of the output neurons
     * Size [N, 2] (N entries, each with 2 activations, one for each neuron in the output layer
     */
    const outputLayerActivationsSet = getAppliedMatrix(
        outputLayerInducedLocalFieldsSet,
        this.outputActivationFunction.activate
    );
    
    /**
     * @var errorsSet
     * @description The set of differences between the desired and the real outputs of each output neuron
     * Size [N, 2] (N entries, each with 2 errors, one for each neuron in the output layer)
     */
    const errorsSet = getErrors(outputLayerActivationsSet, expectedOutputSet);
    
    // (N x 2)
    /**
     * @var outputLayerLocalGradientsSet
     * @description The set of local gradients of the output neurons
     * Size [N, 2] (N entries, each with 2 local gradients, one for each neuron in the output layer)
     */
    const outputLayerLocalGradientsSet = this.calculateOutputLocalGradient(errorsSet, outputLayerInducedLocalFieldsSet);
 
    /**
     * @var outputLayerWeightAdjustmentMatrix
     * @description The weight adjustments between hidden and output layers
     * The weight adjustment ij is obtained by multiplying the activation of
     *    the hidden neuron i with the local gradient of the output neuron j.
     *
     * To achieve in matrix multiplication, the hidden layers activation set is
     *    transposed (from [N, 1+3] to [1+3, N])
     *    and multiplied with the output local gradients [N, 2]
     *
     * Size [1+3, 2], a matrix of total weights (i,j) that connect the 1 bias and 3 neurons (i)
     *    of the hidden layer to the 2 (j) output neurons.
     *
     * Each total weight (i,j) is the sum of all weight adjustments obtained from each of the N entries of the batch
     */
    const outputLayerWeightAdjustmentMatrix = getMatrixMultiplication(
        getTransposedMatrix(hiddenLayerActivationSetPlusBias),
        outputLayerLocalGradientsSet
    );
    
    /**
     * @var averageOutputLayerWeightAdjustmentMatrix
     * @description The average of the total weight adjustments across the N entries between hidden and output layer
     * Size [1+3, 2], a matrix of average weights (i,j) that connect the 1 bias and 3 neurons (i)
     *    of the hidden layer to the 2 (j) output neurons.
     *
     * Each average weight (i,j) is the average weight adjustment across the N entries
     */
    const averageOutputLayerWeightAdjustmentMatrix = getAppliedMatrix(
        outputLayerWeightAdjustmentMatrix,
        d => d / inputSet.length
    );
    
    /**
     * @var hiddenLayerLocalGradientsSet
     * @description The set of local gradients of the hidden layer neurons
     * Size [N, 3] (N entries, each with 3 local gradients, one for each neuron in the hidden layer)
     */
    const hiddenLayerLocalGradientsSet = this.calculateHiddenLayerLocalGradientSet(
        outputLayerLocalGradientsSet,
        hiddenLayerInducedLocalFieldsSet
    );
    
    /**
     * @var hiddenLayerWeightAdjustmentMatrix
     * @description The weight adjustments between input and hidden layers
     * The weight adjustment ij is obtained by multiplying the input i
     *    with the local gradient of the hidden neuron j.
     *
     * To achieve in matrix multiplication, the input set is
     *    transposed (from [N, 1+2] to [1+2, N])
     *    and multiplied with the hidden local gradients [N, 3]
     *
     * Size [1+2, 3], a matrix of total weights (i,j) that connect the 1 bias and 2 neurons (i)
     *    of the input layer to the 3 (j) hidden neurons.
     *
     * Each total weight (i,j) is the sum of all weight adjustments obtained from each of the N entries of the batch
     */
    const hiddenLayerWeightAdjustmentMatrix = getMatrixMultiplication(
        getTransposedMatrix(inputSetPlusBias),
        hiddenLayerLocalGradientsSet
    );
  
    /**
     * @var averageHiddenLayerWeightAdjustmentMatrix
     * @description The average of the total weight adjustments across the N entries between input and hidden layer
     * Size [1+2, 3], a matrix of average weights (i,j) that connect the 1 bias and 2 neurons (i)
     *    of the input layer to the 3 (j) output neurons.
     *
     * Each average weight (i,j) is the average weight adjustment across the N entries
     */
    const averageHiddenLayerWeightAdjustmentMatrix = getAppliedMatrix(
        hiddenLayerWeightAdjustmentMatrix,
        d => d / inputSet.length
    );
    
    // TODO: Update weights with learning rate
  }
  
  /**
   * @method calculateOutputLocalGradient
   * @param errorsSet - The set of errors of the last layer.
   *    Size [N, 2] (N entries, each with 2 errors, one for each neuron in the output layer)
   * @param outputLocalInducedFieldsSet - The set of local induced fields in the output layer
   *    Size [N, 2] (N entries, each with 2 induced local fields, one for each neuron in the output layer)
   *
   * @returns The set of local gradients for output neurons.
   *    Size [N, 2] (N entries, each with 2 local gradients, one for each neuron in the output layer)
   */
  calculateOutputLocalGradient(
      errorsSet: number[][],
      outputLocalInducedFieldsSet: number[][]
  ): any {
  
    /**
     * @var outputLayerDerivativesSet
     * @description The derivatives of the activation function of output neurons at the local induced field
     *    Size [N, 2] (N entries, each with 2 derivatives, one for each output neuron)
     */
    const outputLayerDerivativesSet = getAppliedMatrix(
        outputLocalInducedFieldsSet,
        this.outputActivationFunction.derivative
    );
    
    /**
     * @returns the set of output local gradients
     * Obtained by multiplying the error of each output neuron with the derivative of the same neuron
     *
     * Size [N, 2] (N entries, each with 2 local gradients, one for each output neuron)
     */
    return errorsSet.map((errors, n) => {
      const derivatives = outputLayerDerivativesSet[n];
      return errors.map((errorOfNeuron, i) => {
        const derivativeOfNeuron = derivatives[i];
        return errorOfNeuron * derivativeOfNeuron;
      });
    });
  }
  
  /**
   * @method calculateHiddenLayerLocalGradientSet
   * @param outputLayerLocalGradientsSet - The set of local gradients in the output layer.
   *    Size [N, 2] (N entries, each with 2 local gradients, one for each neuron in the output layer)
   * @param hiddenLayerInducedLocalFieldsSet - The set of induced local fields in the hidden layer.
   *    Size [N, 3] (N entries, each with 3 induced local fields, one for each neuron in the hidden layer)
   * @returns The set of local gradients for the hidden layer.
   *    Size [N, 3] (N entries, each with 3 local gradients, one for each neuron in the hidden layer)
   */
  calculateHiddenLayerLocalGradientSet(
      outputLayerLocalGradientsSet: any,
      hiddenLayerInducedLocalFieldsSet: number[][]
  ): any {
    /**
     * @var hiddenLayerDerivativesSet
     * @description The derivatives of the activation function of hidden neurons at the local induced field
     *    Size [N, 3] (N entries, each with 3 derivatives, one for each hidden neuron)
     */
    const hiddenLayerDerivativesSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenActivationFunction.derivative
    );
    
    /**
     * @var weights2WithoutBias
     * @description The weights connecting the hidden layer to the output layer, but without the bias weights
     * Reason: there is no connection from the input layer to the hidden layer bias,
     *  so there is no need to obtain the local gradient of the hidden layer bias,
     *  because it is not going to be propagated back to any weight in the input layer
     *
     * Size [3, 2] (A matrix of weights (i,j) connecting the 3 (i) neurons of the hidden layer to the
     *    2 (j) output neurons of the output layer.
     */
    const weights2WithoutBias = this.weights2.slice(1)
    
    /**
     * @var backpropagatedGradientsSet
     * @description The propagated gradients from the output neurons back to the hidden layer, weighted by the connections
     * This is not yet the local gradient of the hidden neuron,
     *        since the propagated gradient still needs to be multiplied with the
     *        derivative of the hidden neuron at the local induced field
     *
     * In matrix multiplication, this is obtained by multiplying
     *    the set of local gradients of the output layer [N, 2]
     *    with the transposed weights2WithoutBias (from [3,2] to [2,3])
     *
     * Size [N, 3] (N entries, each with 3 backpropagated gradients from the output to the hidden layer)
     */
    const backpropagatedGradientsSet = getMatrixMultiplication(
        outputLayerLocalGradientsSet,
        getTransposedMatrix(weights2WithoutBias)
    )
  
    /**
     * @returns the set of hidden local gradients
     * Obtained by multiplying the backpropagated gradient of each hidden neuron with the derivative of the same neuron
     *
     * Size [N, 3] (N entries, each with 3 local gradients, one for each hidden neuron)
     */
    return backpropagatedGradientsSet.map((backpropagatedGradients, n) => {
      const derivatives = hiddenLayerDerivativesSet[n];
      return backpropagatedGradients.map((backPropagatedGradientOfNeuron, i) => {
        const derivativeOfNeuron = derivatives[i];
        return backPropagatedGradientOfNeuron * derivativeOfNeuron;
      });
    });
  }
}
