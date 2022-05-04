import {ActivationFunction} from "../activation-function/activation-function.js";
import {Matrix} from "../math/Matrix.js";

/**
 * @class NN
 * @description Encapsulate a MLP with one hidden layer that can be trained using backpropagation
 */
export class NN {
  /**
   * @var this.weights1
   * @description The math of size [1+I,H] of (i,j) weights between the input layer and the hidden layer
   *    Each weight (i,j) connects the 1 bias and the I neurons (i) of the input layer
   *    to the H neurons (j) of the hidden layer
   */
  private weights1: Matrix;
  
  /**
   * @var this.weights2
   * @description The math of size [1+H,O] of (i,j) weights between the hidden layer and the output layer
   *    Each weight (i,j) connects the 1 bias and the H neurons (i) of the hidden layer
   *    to the O neurons (j) of the output layer
   */
  private weights2: Matrix;
  
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
    this.weights1 = new Matrix(1 + this.inputNeurons, this.hiddenNeurons, true);
    this.weights2 = new Matrix(1 + this.hiddenNeurons, this.outputNeurons, true);
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
    const inputSetPlusBias = Matrix.from(inputSet.map(inputs => [1, ...inputs]));
    
    /**
     * @var inducedLocalFieldsSetOfHiddenLayer
     * @description The set of induced local fields for the hidden layer
     * Multiplies inputSetPlusBias [N, 1+I] with this.weights1 [1+I, H],
     * Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
     */
    const inducedLocalFieldsSetOfHiddenLayer = inputSetPlusBias.leftMultiplyWith(this.weights1);
    
    /**
     * @var hiddenLayerActivationSet
     * @description The set of activations for the hidden layer
     * Applies the hidden layer activation function to the induced local fields of the hidden neurons
     * Size [N, H] (N entries, each with H activations, one for each neuron in the hidden layer
     */
    const activationsSetOfHiddenLayer = inducedLocalFieldsSetOfHiddenLayer.mapValues(
        this.hiddenLayerActivationFunction.activate
    );
    
    /**
     * @var activationSetPlusBiasOfHiddenLayer
     * @description The activationsSetOfHiddenLayer [N, H], pre-pended with a bias [N, 1+H]
     * Size [N, 1+H] (N entries, each with 1 fixed bias and H activations, one for each neuron in the hidden layer)
     */
    const activationSetPlusBiasOfHiddenLayer = activationsSetOfHiddenLayer.mapRows(activations => [1, ...activations]);
    
    
    /**
     * @var inducedLocalFieldsSetOfOutputLayer
     * @description The set of induced local fields for the output layer
     * Multiplies activationSetPlusBiasOfHiddenLayer [N, 1+H] with this.weights2 [1+H, O],
     * Size [N, O] (N entries, each with O induced local fields, one for each neuron in the output layer)
     */
    const inducedLocalFieldsSetOfOutputLayer = activationSetPlusBiasOfHiddenLayer.leftMultiplyWith(this.weights2);
    
    /**
     * @var activationsSetOfOutputLayer
     * @description The set of activations for the output layer
     * Applies the output layer activation function to the induced local fields of the output neurons
     * Size [N, O] (N entries, each with O activations, one for each neuron in the output layer
     */
    const activationsSetOfOutputLayer = inducedLocalFieldsSetOfOutputLayer.mapValues(this.outputLayerActivationFunction.activate);
    
    /**
     * @var errorsSet
     * @description The set of differences between the desired and the real outputNeurons of each output neuron
     * Size [N, O] (N entries, each with O errors, one for each neuron in the output layer)
     */
    const errorsSet = Matrix.from(expectedOutputSet)
        .operateWith(activationsSetOfOutputLayer, (e, o) => e - o);
    
    /**
     * @var localGradientsSetOfOutputLayer
     * @description The set of local gradients of the output neurons
     * Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
     */
    const localGradientsSetOfOutputLayer = this.calculateOutputLocalGradient(
        errorsSet,
        inducedLocalFieldsSetOfOutputLayer
    );
    
    /**
     * @var totalWeightAdjustmentMatrixOfOutputLayer
     * @description The weight adjustments between hidden and output layers
     * The weight adjustment ij is obtained by multiplying the activation of
     *    the hidden neuron i with the local gradient of the output neuron j.
     *
     * To achieve in math multiplication, the hidden layers activation set is
     *    transposed (from [N, 1+H] to [1+H, N])
     *    and multiplied with the output local gradients [N, O]
     *
     * Size [1+H, O], a math of total weights (i,j) that connect the 1 bias and H neurons (i)
     *    of the hidden layer to the O (j) output neurons.
     *
     * Each total weight (i,j) is the sum of all weight adjustments obtained from each of the N entries of the batch
     */
    const totalWeightAdjustmentMatrixOfOutputLayer = activationSetPlusBiasOfHiddenLayer.transposed()
        .leftMultiplyWith(localGradientsSetOfOutputLayer);
    
    /**
     * @var averageWeightAdjustmentMatrixOfOutputLayer
     * @description The average of the total weight adjustments across the N entries between hidden and output layer
     * Size [1+H, O], a math of average weights (i,j) that connect the 1 bias and H neurons (i)
     *    of the hidden layer to the O (j) output neurons.
     *
     * Each average weight (i,j) is the average weight adjustment across the N entries
     */
    const averageWeightAdjustmentMatrixOfOutputLayer = totalWeightAdjustmentMatrixOfOutputLayer.mapValues(
        d => d / inputSet.length
    );
    
    /**
     * @var localGradientsSetOfHiddenLayer
     * @description The set of local gradients of the hidden layer neurons
     * Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
     */
    const localGradientsSetOfHiddenLayer = this.calculateHiddenLayerLocalGradientSet(
        localGradientsSetOfOutputLayer,
        inducedLocalFieldsSetOfHiddenLayer
    );
    
    /**
     * @var totalWeightAdjustmentMatrixOfHiddenLayer
     * @description The weight adjustments between input and hidden layers
     * The weight adjustment ij is obtained by multiplying the input i
     *    with the local gradient of the hidden neuron j.
     *
     * To achieve in math multiplication, the input set is
     *    transposed (from [N, 1+I] to [1+I, N])
     *    and multiplied with the hidden local gradients [N, H]
     *
     * Size [1+I, H], a math of total weights (i,j) that connect the 1 bias and I neurons (i)
     *    of the input layer to the H (j) hidden neurons.
     *
     * Each total weight (i,j) is the sum of all weight adjustments obtained from each of the N entries of the batch
     */
    const totalWeightAdjustmentMatrixOfHiddenLayer = inputSetPlusBias.transposed()
        .leftMultiplyWith(localGradientsSetOfHiddenLayer);
    
    /**
     * @var averageWeightAdjustmentMatrixOfHiddenLayer
     * @description The average of the total weight adjustments across the N entries between input and hidden layer
     * Size [1+I, H], a math of average weights (i,j) that connect the 1 bias and I neurons (i)
     *    of the input layer to the H (j) hidden neurons.
     *
     * Each average weight (i,j) is the average weight adjustment across the N entries
     */
    const averageWeightAdjustmentMatrixOfHiddenLayer = totalWeightAdjustmentMatrixOfHiddenLayer.mapValues(
        d => d / inputSet.length
    );
    
    
    this.weights1 = this.weights1.operateWith(
        averageWeightAdjustmentMatrixOfHiddenLayer,
        (weight, newWeight) => weight + this.learningRate * newWeight
    );
    
    this.weights2 = this.weights2.operateWith(
        averageWeightAdjustmentMatrixOfOutputLayer,
        (weight, newWeight) => weight + this.learningRate * newWeight
    );
  }
  
  /**
   * @method calculateOutputLocalGradient
   * @param errorsSet - The set of errors of the last layer.
   *    Size [N, O] (N entries, each with O errors, one for each neuron in the output layer)
   * @param localInducedFieldsSetOfOutputLayer - The set of local induced fields in the output layer
   *    Size [N, O] (N entries, each with O induced local fields, one for each neuron in the output layer)
   *
   * @returns The set of local gradients for output neurons.
   *    Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
   */
  private calculateOutputLocalGradient(
      errorsSet: Matrix,
      localInducedFieldsSetOfOutputLayer: Matrix
  ): Matrix {
    
    /**
     * @var derivativesSetOfOutputLayer
     * @description The derivatives of the activation function of output neurons at the local induced field
     *    Size [N, O] (N entries, each with O derivatives, one for each output neuron)
     */
    const derivativesSetOfOutputLayer = localInducedFieldsSetOfOutputLayer.mapValues(this.outputLayerActivationFunction.derivative);
    
    /**
     * @returns the set of output local gradients
     * Obtained by multiplying the error of each output neuron with the derivative of the same neuron
     *
     * Size [N, O] (N entries, each with O local gradients, one for each output neuron)
     */
    return errorsSet.operateWith(derivativesSetOfOutputLayer, (e, d) => e * d);
  }
  
  /**
   * @method calculateHiddenLayerLocalGradientSet
   * @param localGradientsSetOfOutputLayer - The set of local gradients in the output layer.
   *    Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
   * @param inducedLocalFieldsSetOfHiddenLayer - The set of induced local fields in the hidden layer.
   *    Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
   * @returns The set of local gradients for the hidden layer.
   *    Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
   */
  private calculateHiddenLayerLocalGradientSet(
      localGradientsSetOfOutputLayer: Matrix,
      inducedLocalFieldsSetOfHiddenLayer: Matrix
  ): Matrix {
    /**
     * @var hiddenLayerDerivativesSet
     * @description The derivatives of the activation function of hidden neurons at the local induced field
     *    Size [N, H] (N entries, each with H derivatives, one for each hidden neuron)
     */
    const hiddenLayerDerivativesSet = inducedLocalFieldsSetOfHiddenLayer.mapValues(
        this.hiddenLayerActivationFunction.derivative
    );
    
    /**
     * @var weights2WithoutBias
     * @description The weights connecting the hidden layer to the output layer, but without the bias weights
     * Reason: there is no connection from the input layer to the hidden layer bias,
     *  so there is no need to obtain the local gradient of the hidden layer bias,
     *  because it is not going to be propagated back to any weight in the input layer
     *
     * Size [H, O] (A math of weights (i,j) connecting the H (i) neurons of the hidden layer to the
     *    O (j) output neurons of the output layer.
     */
    const weights2WithoutBias = this.weights2.sliceRows(1);
    
    /**
     * @var backpropagatedGradientsSet
     * @description The propagated gradients from the output neurons back to the hidden layer, weighted by the connections
     * This is not yet the local gradient of the hidden neuron,
     *        since the propagated gradient still needs to be multiplied with the
     *        derivative of the hidden neuron at the local induced field
     *
     * In math multiplication, this is obtained by multiplying
     *    the set of local gradients of the output layer [N, O]
     *    with the transposed weights2WithoutBias (from [H,O] to [O,H])
     *
     * Size [N, H] (N entries, each with H backpropagated gradients from the output to the hidden layer)
     */
    const backpropagatedGradientsSet = localGradientsSetOfOutputLayer
        .leftMultiplyWith(weights2WithoutBias.transposed());
    
    /**
     * @returns the set of hidden local gradients
     * Obtained by multiplying the backpropagated gradient of each hidden neuron with the derivative of the same neuron
     *
     * Size [N, H] (N entries, each with H local gradients, one for each hidden neuron)
     */
    return backpropagatedGradientsSet.operateWith(hiddenLayerDerivativesSet, (b, h) => b * h);
  }
}
