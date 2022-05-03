var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
import { getMatrixMultiplication } from "../hooks/get-matrix-multiplication.js";
import { getAppliedMatrix } from "../hooks/get-applied-matrix.js";
import { getErrors } from "../hooks/get-errors.js";
import { getTransposedMatrix } from "../hooks/get-transposed-matrix.js";
/**
 * @class NN
 * @description Encapsulate a MLP with one hidden layer that can be trained using backpropagation
 */
var NN = /** @class */ (function () {
    /**
     * @constructor
     * @param inputNeurons - The number of input neurons
     * @param hiddenNeurons - The number of hidden neurons
     * @param hiddenLayerActivationFunction - The activation function of hidden neurons
     * @param outputNeurons - The number of output neurons
     * @param outputLayerActivationFunction - The activation function of output neurons
     * @param learningRate - The learning rate of weight update
     */
    function NN(inputNeurons, hiddenNeurons, hiddenLayerActivationFunction, outputNeurons, outputLayerActivationFunction, learningRate) {
        var _this = this;
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.hiddenLayerActivationFunction = hiddenLayerActivationFunction;
        this.outputNeurons = outputNeurons;
        this.outputLayerActivationFunction = outputLayerActivationFunction;
        this.learningRate = learningRate;
        this.weights1 =
            new Array(1 + this.inputNeurons).fill(0).map(function () { return new Array(_this.hiddenNeurons).fill(0).map(function () { return Math.random(); }); });
        this.weights2 =
            new Array(1 + this.hiddenNeurons + 1).fill(0).map(function () { return new Array(_this.outputNeurons).fill(0).map(function () { return Math.random(); }); });
    }
    /**
     * @description a weight update with the provided batch
     * @param inputSet - The batch to use for weight update
     * @param expectedOutputSet - The expected output of neurons for each entry in the batch
     */
    NN.prototype.train = function (inputSet, expectedOutputSet) {
        /**
         * @var inputSetPlusBias
         * @description The input set [N, I], pre-pended with a bias [N, 1+I]
         * Size [N, 1+I] (N entries, each with 1 fixed bias and I inputNeurons)
         */
        var inputSetPlusBias = inputSet.map(function (inputs) { return __spreadArray([1], inputs, true); });
        /**
         * @var hiddenLayerInducedLocalFieldsSet
         * @description The set of induced local fields for the hidden layer
         * Multiplies inputSetPlusBias [N, 1+I] with this.weights1 [1+I, H],
         * Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
         */
        var hiddenLayerInducedLocalFieldsSet = getMatrixMultiplication(inputSetPlusBias, this.weights1);
        /**
         * @var hiddenLayerActivationSet
         * @description The set of activations for the hidden layer
         * Applies the hidden layer activation function to the induced local fields of the hidden neurons
         * Size [N, H] (N entries, each with H activations, one for each neuron in the hidden layer
         */
        var hiddenLayerActivationsSet = getAppliedMatrix(hiddenLayerInducedLocalFieldsSet, this.hiddenLayerActivationFunction.activate);
        /**
         * @var hiddenLayerActivationSetPlusBias
         * @description The hiddenLayerActivationsSet [N, H], pre-pended with a bias [N, 1+H]
         * Size [N, 1+H] (N entries, each with 1 fixed bias and H activations, one for each neuron in the hidden layer)
         */
        var hiddenLayerActivationSetPlusBias = hiddenLayerActivationsSet.map(function (a) { return __spreadArray([1], a, true); });
        /**
         * @var outputLayerInducedLocalFieldsSet
         * @description The set of induced local fields for the output layer
         * Multiplies hiddenLayerActivationSetPlusBias [N, 1+H] with this.weights2 [1+H, O],
         * Size [N, O] (N entries, each with O induced local fields, one for each neuron in the output layer)
         */
        var outputLayerInducedLocalFieldsSet = getMatrixMultiplication(hiddenLayerActivationSetPlusBias, this.weights2);
        /**
         * @var outputLayerActivationsSet
         * @description The set of activations for the output layer
         * Applies the output layer activation function to the induced local fields of the output neurons
         * Size [N, O] (N entries, each with O activations, one for each neuron in the output layer
         */
        var outputLayerActivationsSet = getAppliedMatrix(outputLayerInducedLocalFieldsSet, this.outputLayerActivationFunction.activate);
        /**
         * @var errorsSet
         * @description The set of differences between the desired and the real outputNeurons of each output neuron
         * Size [N, O] (N entries, each with O errors, one for each neuron in the output layer)
         */
        var errorsSet = getErrors(outputLayerActivationsSet, expectedOutputSet);
        /**
         * @var outputLayerLocalGradientsSet
         * @description The set of local gradients of the output neurons
         * Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
         */
        var outputLayerLocalGradientsSet = this.calculateOutputLocalGradient(errorsSet, outputLayerInducedLocalFieldsSet);
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
        var outputLayerWeightAdjustmentMatrix = getMatrixMultiplication(getTransposedMatrix(hiddenLayerActivationSetPlusBias), outputLayerLocalGradientsSet);
        /**
         * @var averageOutputLayerWeightAdjustmentMatrix
         * @description The average of the total weight adjustments across the N entries between hidden and output layer
         * Size [1+H, O], a matrix of average weights (i,j) that connect the 1 bias and H neurons (i)
         *    of the hidden layer to the O (j) output neurons.
         *
         * Each average weight (i,j) is the average weight adjustment across the N entries
         */
        var averageOutputLayerWeightAdjustmentMatrix = getAppliedMatrix(outputLayerWeightAdjustmentMatrix, function (d) { return d / inputSet.length; });
        /**
         * @var hiddenLayerLocalGradientsSet
         * @description The set of local gradients of the hidden layer neurons
         * Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
         */
        var hiddenLayerLocalGradientsSet = this.calculateHiddenLayerLocalGradientSet(outputLayerLocalGradientsSet, hiddenLayerInducedLocalFieldsSet);
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
        var hiddenLayerWeightAdjustmentMatrix = getMatrixMultiplication(getTransposedMatrix(inputSetPlusBias), hiddenLayerLocalGradientsSet);
        /**
         * @var averageHiddenLayerWeightAdjustmentMatrix
         * @description The average of the total weight adjustments across the N entries between input and hidden layer
         * Size [1+I, H], a matrix of average weights (i,j) that connect the 1 bias and I neurons (i)
         *    of the input layer to the H (j) hidden neurons.
         *
         * Each average weight (i,j) is the average weight adjustment across the N entries
         */
        var averageHiddenLayerWeightAdjustmentMatrix = getAppliedMatrix(hiddenLayerWeightAdjustmentMatrix, function (d) { return d / inputSet.length; });
        // TODO: Update weights with learning rate
    };
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
    NN.prototype.calculateOutputLocalGradient = function (errorsSet, outputLocalInducedFieldsSet) {
        /**
         * @var outputLayerDerivativesSet
         * @description The derivatives of the activation function of output neurons at the local induced field
         *    Size [N, O] (N entries, each with O derivatives, one for each output neuron)
         */
        var outputLayerDerivativesSet = getAppliedMatrix(outputLocalInducedFieldsSet, this.outputLayerActivationFunction.derivative);
        /**
         * @returns the set of output local gradients
         * Obtained by multiplying the error of each output neuron with the derivative of the same neuron
         *
         * Size [N, O] (N entries, each with O local gradients, one for each output neuron)
         */
        return errorsSet.map(function (errors, n) {
            var derivatives = outputLayerDerivativesSet[n];
            return errors.map(function (errorOfNeuron, i) {
                var derivativeOfNeuron = derivatives[i];
                return errorOfNeuron * derivativeOfNeuron;
            });
        });
    };
    /**
     * @method calculateHiddenLayerLocalGradientSet
     * @param outputLayerLocalGradientsSet - The set of local gradients in the output layer.
     *    Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
     * @param hiddenLayerInducedLocalFieldsSet - The set of induced local fields in the hidden layer.
     *    Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
     * @returns The set of local gradients for the hidden layer.
     *    Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
     */
    NN.prototype.calculateHiddenLayerLocalGradientSet = function (outputLayerLocalGradientsSet, hiddenLayerInducedLocalFieldsSet) {
        /**
         * @var hiddenLayerDerivativesSet
         * @description The derivatives of the activation function of hidden neurons at the local induced field
         *    Size [N, H] (N entries, each with H derivatives, one for each hidden neuron)
         */
        var hiddenLayerDerivativesSet = getAppliedMatrix(hiddenLayerInducedLocalFieldsSet, this.hiddenLayerActivationFunction.derivative);
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
        var weights2WithoutBias = this.weights2.slice(1);
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
        var backpropagatedGradientsSet = getMatrixMultiplication(outputLayerLocalGradientsSet, getTransposedMatrix(weights2WithoutBias));
        /**
         * @returns the set of hidden local gradients
         * Obtained by multiplying the backpropagated gradient of each hidden neuron with the derivative of the same neuron
         *
         * Size [N, H] (N entries, each with H local gradients, one for each hidden neuron)
         */
        return backpropagatedGradientsSet.map(function (backpropagatedGradients, n) {
            var derivatives = hiddenLayerDerivativesSet[n];
            return backpropagatedGradients.map(function (backPropagatedGradientOfNeuron, i) {
                var derivativeOfNeuron = derivatives[i];
                return backPropagatedGradientOfNeuron * derivativeOfNeuron;
            });
        });
    };
    return NN;
}());
export { NN };
//# sourceMappingURL=nn.js.map