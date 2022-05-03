import {ActivationFunction} from "../activation-function/activation-function.js";
import {getMatrixMultiplication} from "../hooks/get-matrix-multiplication.js";
import {getAppliedMatrix} from "../hooks/get-applied-matrix.js";
import {getMatrixOperation} from "../hooks/get-matrix-operation.js";
import {getTransposedMatrix} from "../hooks/get-transposed-matrix.js";
import {getMatrix} from "../hooks/get-matrix.js";

/**
 * @class NN
 * @description Encapsulate a MLP with one hidden layer that can be trained using backpropagation
 */
export class NN {
  /**
   * @var this.weights1
   * @description The matrix of size [1+I,H] of (i,j) weights between the input layer and the hidden layer
   *    Each weight (i,j) connects the 1 bias and the I neurons (i) of the input layer
   *    to the H neurons (j) of the hidden layer
   */
  private readonly weights1: number[][];
  
  /**
   * @var this.weights2
   * @description The matrix of size [1+H,O] of (i,j) weights between the hidden layer and the output layer
   *    Each weight (i,j) connects the 1 bias and the H neurons (i) of the hidden layer
   *    to the O neurons (j) of the output layer
   */
  private readonly weights2: number[][];
  
  /**
   * @constructor
   * @param inputNeurons - The number of input neurons
   * @param hiddenNeurons - The number of hidden neurons
   * @param hiddenLayerActivationFunction - The activation function of hidden neurons
   * @param outputNeurons - The number of output neurons
   * @param outputLayerActivationFunction - The activation function of output neurons
   * @param learningRate - The learning rate of weight update
   */
  constructor(
      private inputNeurons: number,
      private hiddenNeurons: number,
      private hiddenLayerActivationFunction: ActivationFunction,
      private outputNeurons: number,
      private outputLayerActivationFunction: ActivationFunction,
      private learningRate: number
  ) {
    this.weights1 = getAppliedMatrix(getMatrix(1 + this.inputNeurons, this.hiddenNeurons), () => Math.random());
    this.weights2 = getAppliedMatrix(getMatrix(1 + this.hiddenNeurons, this.outputNeurons), () => Math.random());
  }
  
  /**
   * @description a weight update with the provided batch
   * @param inputSet - The batch to use for weight update
   * @param expectedOutputSet - The expected output of neurons for each entry in the batch
   */
  train(inputSet: number[][], expectedOutputSet: number[][]): void {
    /**
     * @var inputSetPlusBias
     * @description The input set [N, I], pre-pended with a bias [N, 1+I]
     * Size [N, 1+I] (N entries, each with 1 fixed bias and I inputNeurons)
     */
    const inputSetPlusBias = inputSet.map(inputs => [1, ...inputs]);
    
    /**
     * @var hiddenLayerInducedLocalFieldsSet
     * @description The set of induced local fields for the hidden layer
     * Multiplies inputSetPlusBias [N, 1+I] with this.weights1 [1+I, H],
     * Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
     */
    const hiddenLayerInducedLocalFieldsSet = getMatrixMultiplication(inputSetPlusBias, this.weights1);
    
    /**
     * @var hiddenLayerActivationSet
     * @description The set of activations for the hidden layer
     * Applies the hidden layer activation function to the induced local fields of the hidden neurons
     * Size [N, H] (N entries, each with H activations, one for each neuron in the hidden layer
     */
    const hiddenLayerActivationsSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenLayerActivationFunction.activate
    );
    
    /**
     * @var hiddenLayerActivationSetPlusBias
     * @description The hiddenLayerActivationsSet [N, H], pre-pended with a bias [N, 1+H]
     * Size [N, 1+H] (N entries, each with 1 fixed bias and H activations, one for each neuron in the hidden layer)
     */
    const hiddenLayerActivationSetPlusBias = hiddenLayerActivationsSet.map(a => [1, ...a]);
    
    
    /**
     * @var outputLayerInducedLocalFieldsSet
     * @description The set of induced local fields for the output layer
     * Multiplies hiddenLayerActivationSetPlusBias [N, 1+H] with this.weights2 [1+H, O],
     * Size [N, O] (N entries, each with O induced local fields, one for each neuron in the output layer)
     */
    const outputLayerInducedLocalFieldsSet = getMatrixMultiplication(
        hiddenLayerActivationSetPlusBias,
        this.weights2
    );
    
    /**
     * @var outputLayerActivationsSet
     * @description The set of activations for the output layer
     * Applies the output layer activation function to the induced local fields of the output neurons
     * Size [N, O] (N entries, each with O activations, one for each neuron in the output layer
     */
    const outputLayerActivationsSet = getAppliedMatrix(
        outputLayerInducedLocalFieldsSet,
        this.outputLayerActivationFunction.activate
    );
    
    /**
     * @var errorsSet
     * @description The set of differences between the desired and the real outputNeurons of each output neuron
     * Size [N, O] (N entries, each with O errors, one for each neuron in the output layer)
     */
    const errorsSet = getMatrixOperation(
        expectedOutputSet,
        outputLayerActivationsSet,
        (e, o) => e - o
    );
    
    /**
     * @var outputLayerLocalGradientsSet
     * @description The set of local gradients of the output neurons
     * Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
     */
    const outputLayerLocalGradientsSet = this.calculateOutputLocalGradient(errorsSet, outputLayerInducedLocalFieldsSet);
    
    /**
     * @var outputLayerWeightAdjustmentMatrix
     * @description The weight adjustments between hidden and output layers
     * The weight adjustment ij is obtained by multiplying the activation of
     *    the hidden neuron i with the local gradient of the output neuron j.
     *
     * To achieve in matrix multiplication, the hidden layers activation set is
     *    transposed (from [N, 1+H] to [1+H, N])
     *    and multiplied with the output local gradients [N, O]
     *
     * Size [1+H, O], a matrix of total weights (i,j) that connect the 1 bias and H neurons (i)
     *    of the hidden layer to the O (j) output neurons.
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
     * Size [1+H, O], a matrix of average weights (i,j) that connect the 1 bias and H neurons (i)
     *    of the hidden layer to the O (j) output neurons.
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
     * Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
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
     *    transposed (from [N, 1+I] to [1+I, N])
     *    and multiplied with the hidden local gradients [N, H]
     *
     * Size [1+I, H], a matrix of total weights (i,j) that connect the 1 bias and I neurons (i)
     *    of the input layer to the H (j) hidden neurons.
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
     * Size [1+I, H], a matrix of average weights (i,j) that connect the 1 bias and I neurons (i)
     *    of the input layer to the H (j) hidden neurons.
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
   *    Size [N, O] (N entries, each with O errors, one for each neuron in the output layer)
   * @param outputLocalInducedFieldsSet - The set of local induced fields in the output layer
   *    Size [N, O] (N entries, each with O induced local fields, one for each neuron in the output layer)
   *
   * @returns The set of local gradients for output neurons.
   *    Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
   */
  private calculateOutputLocalGradient(
      errorsSet: number[][],
      outputLocalInducedFieldsSet: number[][]
  ): any {
    
    /**
     * @var outputLayerDerivativesSet
     * @description The derivatives of the activation function of output neurons at the local induced field
     *    Size [N, O] (N entries, each with O derivatives, one for each output neuron)
     */
    const outputLayerDerivativesSet = getAppliedMatrix(
        outputLocalInducedFieldsSet,
        this.outputLayerActivationFunction.derivative
    );
    
    /**
     * @returns the set of output local gradients
     * Obtained by multiplying the error of each output neuron with the derivative of the same neuron
     *
     * Size [N, O] (N entries, each with O local gradients, one for each output neuron)
     */
    return getMatrixOperation(errorsSet, outputLayerDerivativesSet, (e, o) => e * o);
  }
  
  /**
   * @method calculateHiddenLayerLocalGradientSet
   * @param outputLayerLocalGradientsSet - The set of local gradients in the output layer.
   *    Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
   * @param hiddenLayerInducedLocalFieldsSet - The set of induced local fields in the hidden layer.
   *    Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
   * @returns The set of local gradients for the hidden layer.
   *    Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
   */
  private calculateHiddenLayerLocalGradientSet(
      outputLayerLocalGradientsSet: any,
      hiddenLayerInducedLocalFieldsSet: number[][]
  ): any {
    /**
     * @var hiddenLayerDerivativesSet
     * @description The derivatives of the activation function of hidden neurons at the local induced field
     *    Size [N, H] (N entries, each with H derivatives, one for each hidden neuron)
     */
    const hiddenLayerDerivativesSet = getAppliedMatrix(
        hiddenLayerInducedLocalFieldsSet,
        this.hiddenLayerActivationFunction.derivative
    );
    
    /**
     * @var weights2WithoutBias
     * @description The weights connecting the hidden layer to the output layer, but without the bias weights
     * Reason: there is no connection from the input layer to the hidden layer bias,
     *  so there is no need to obtain the local gradient of the hidden layer bias,
     *  because it is not going to be propagated back to any weight in the input layer
     *
     * Size [H, O] (A matrix of weights (i,j) connecting the H (i) neurons of the hidden layer to the
     *    O (j) output neurons of the output layer.
     */
    const weights2WithoutBias = this.weights2.slice(1);
    
    /**
     * @var backpropagatedGradientsSet
     * @description The propagated gradients from the output neurons back to the hidden layer, weighted by the connections
     * This is not yet the local gradient of the hidden neuron,
     *        since the propagated gradient still needs to be multiplied with the
     *        derivative of the hidden neuron at the local induced field
     *
     * In matrix multiplication, this is obtained by multiplying
     *    the set of local gradients of the output layer [N, O]
     *    with the transposed weights2WithoutBias (from [H,O] to [O,H])
     *
     * Size [N, H] (N entries, each with H backpropagated gradients from the output to the hidden layer)
     */
    const backpropagatedGradientsSet = getMatrixMultiplication(
        outputLayerLocalGradientsSet,
        getTransposedMatrix(weights2WithoutBias)
    );
    
    /**
     * @returns the set of hidden local gradients
     * Obtained by multiplying the backpropagated gradient of each hidden neuron with the derivative of the same neuron
     *
     * Size [N, H] (N entries, each with H local gradients, one for each hidden neuron)
     */
    return getMatrixOperation(backpropagatedGradientsSet, hiddenLayerDerivativesSet, (b, h) => b * h);
  }
}
