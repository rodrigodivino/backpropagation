import { Matrix } from "../math/Matrix.js";
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
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.hiddenLayerActivationFunction = hiddenLayerActivationFunction;
        this.outputNeurons = outputNeurons;
        this.outputLayerActivationFunction = outputLayerActivationFunction;
        this.learningRate = learningRate;
        this.weights1 = new Matrix(this.inputNeurons, this.hiddenNeurons, true);
        this.weights2 = new Matrix(this.hiddenNeurons, this.outputNeurons, true);
        this.bias1 = new Array(this.hiddenNeurons).fill(0).map(function () { return Math.random(); });
        this.bias2 = new Array(this.outputNeurons).fill(0).map(function () { return Math.random(); });
    }
    /**
     * @description a weight update with the provided batch
     * @param inputSet - The batch to use for weight update
     * @param expectedOutputSet - The expected output of neurons for each entry in the batch
     */
    NN.prototype.train = function (rawInputSet, rawExpectedOutputSet) {
        var _this = this;
        var inputSet = Matrix.from(rawInputSet).transposed();
        var expectedOutputSet = Matrix.from(rawExpectedOutputSet).transposed();
        /**
         * @var inducedLocalFieldsSetOfHiddenLayer
         * @description The set of induced local fields for the hidden layer
         * Multiplies inputSetPlusBias [N, 1+I] with this.weights1 [1+I, H],
         * Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
         */
        var inducedLocalFieldsSetOfHiddenLayer = this.weights1.transposed()
            .leftMultiplyWith(inputSet)
            .mapColumns(function (column) { return column.map(function (value, c) { return value + _this.bias1[c]; }); });
        /**
         * @var hiddenLayerActivationSet
         * @description The set of activations for the hidden layer
         * Applies the hidden layer activation function to the induced local fields of the hidden neurons
         * Size [N, H] (N entries, each with H activations, one for each neuron in the hidden layer
         */
        var activationsSetOfHiddenLayer = inducedLocalFieldsSetOfHiddenLayer
            .mapValues(this.hiddenLayerActivationFunction.activate);
        /**
         * @var inducedLocalFieldsSetOfOutputLayer
         * @description The set of induced local fields for the output layer
         * Multiplies activationSetPlusBiasOfHiddenLayer [N, 1+H] with this.weights2 [1+H, O],
         * Size [N, O] (N entries, each with O induced local fields, one for each neuron in the output layer)
         */
        var inducedLocalFieldsSetOfOutputLayer = this.weights2.transposed()
            .leftMultiplyWith(activationsSetOfHiddenLayer)
            .mapColumns(function (column) { return column.map(function (value, c) { return value + _this.bias2[c]; }); });
        /**
         * @var activationsSetOfOutputLayer
         * @description The set of activations for the output layer
         * Applies the output layer activation function to the induced local fields of the output neurons
         * Size [N, O] (N entries, each with O activations, one for each neuron in the output layer
         */
        var activationsSetOfOutputLayer = inducedLocalFieldsSetOfOutputLayer
            .mapValues(this.outputLayerActivationFunction.activate);
        /**
         * @var errorsSet
         * @description The set of differences between the desired and the real outputNeurons of each output neuron
         * Size [N, O] (N entries, each with O errors, one for each neuron in the output layer)
         */
        var errorsSet = expectedOutputSet
            .operateWith(activationsSetOfOutputLayer, function (e, o) { return e - o; });
        console.log("activationsSetOfOutputLayer.data", activationsSetOfOutputLayer.data.join(', '));
        /**
         * @var localGradientsSetOfOutputLayer
         * @description The set of local gradients of the output neurons
         * Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
         */
        var localGradientsSetOfOutputLayer = this.calculateOutputLocalGradient(errorsSet, inducedLocalFieldsSetOfOutputLayer);
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
        var totalWeightAdjustmentMatrixOfOutputLayer = activationsSetOfHiddenLayer
            .leftMultiplyWith(localGradientsSetOfOutputLayer.transposed());
        var totalBias2AdjustmentArray = localGradientsSetOfOutputLayer.mapRows(function (row) { return [row.reduce(function (p, c) { return p + c; }, 0)]; }).transposed().data[0];
        /**
         * @var averageWeightAdjustmentMatrixOfOutputLayer
         * @description The average of the total weight adjustments across the N entries between hidden and output layer
         * Size [1+H, O], a math of average weights (i,j) that connect the 1 bias and H neurons (i)
         *    of the hidden layer to the O (j) output neurons.
         *
         * Each average weight (i,j) is the average weight adjustment across the N entries
         */
        var averageWeightAdjustmentMatrixOfOutputLayer = totalWeightAdjustmentMatrixOfOutputLayer.mapValues(function (d) { return d / inputSet.columns; });
        var averageBias2AdjustmentArray = totalBias2AdjustmentArray.map(function (b) { return b / inputSet.columns; });
        /**
         * @var localGradientsSetOfHiddenLayer
         * @description The set of local gradients of the hidden layer neurons
         * Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
         */
        var localGradientsSetOfHiddenLayer = this.calculateHiddenLayerLocalGradientSet(localGradientsSetOfOutputLayer, inducedLocalFieldsSetOfHiddenLayer);
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
        var totalWeightAdjustmentMatrixOfHiddenLayer = inputSet
            .leftMultiplyWith(localGradientsSetOfHiddenLayer.transposed());
        var totalBias1AdjustmentArray = localGradientsSetOfHiddenLayer
            .mapRows(function (row) { return [row.reduce(function (p, c) { return p + c; }, 0)]; })
            .transposed().data[0];
        /**
         * @var averageWeightAdjustmentMatrixOfHiddenLayer
         * @description The average of the total weight adjustments across the N entries between input and hidden layer
         * Size [1+I, H], a math of average weights (i,j) that connect the 1 bias and I neurons (i)
         *    of the input layer to the H (j) hidden neurons.
         *
         * Each average weight (i,j) is the average weight adjustment across the N entries
         */
        var averageWeightAdjustmentMatrixOfHiddenLayer = totalWeightAdjustmentMatrixOfHiddenLayer.mapValues(function (d) { return d / inputSet.columns; });
        var averageBias1AdjustmentArray = totalBias1AdjustmentArray.map(function (b) { return b / inputSet.columns; });
        // debugger;
        this.weights1 = this.weights1.operateWith(averageWeightAdjustmentMatrixOfHiddenLayer, function (weight, newWeight) { return weight + _this.learningRate * newWeight; });
        this.weights2 = this.weights2.operateWith(averageWeightAdjustmentMatrixOfOutputLayer, function (weight, newWeight) { return weight + _this.learningRate * newWeight; });
        this.bias1 = this.bias1.map(function (b, i) { return b + _this.learningRate * averageBias1AdjustmentArray[i]; });
        this.bias2 = this.bias2.map(function (b, i) { return b + _this.learningRate * averageBias2AdjustmentArray[i]; });
        // debugger;
    };
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
    NN.prototype.calculateOutputLocalGradient = function (errorsSet, localInducedFieldsSetOfOutputLayer) {
        /**
         * @var derivativesSetOfOutputLayer
         * @description The derivatives of the activation function of output neurons at the local induced field
         *    Size [N, O] (N entries, each with O derivatives, one for each output neuron)
         */
        var derivativesSetOfOutputLayer = localInducedFieldsSetOfOutputLayer.mapValues(this.outputLayerActivationFunction.derivative);
        /**
         * @returns the set of output local gradients
         * Obtained by multiplying the error of each output neuron with the derivative of the same neuron
         *
         * Size [N, O] (N entries, each with O local gradients, one for each output neuron)
         */
        return errorsSet.operateWith(derivativesSetOfOutputLayer, function (e, d) { return e * d; });
    };
    /**
     * @method calculateHiddenLayerLocalGradientSet
     * @param localGradientsSetOfOutputLayer - The set of local gradients in the output layer.
     *    Size [N, O] (N entries, each with O local gradients, one for each neuron in the output layer)
     * @param inducedLocalFieldsSetOfHiddenLayer - The set of induced local fields in the hidden layer.
     *    Size [N, H] (N entries, each with H induced local fields, one for each neuron in the hidden layer)
     * @returns The set of local gradients for the hidden layer.
     *    Size [N, H] (N entries, each with H local gradients, one for each neuron in the hidden layer)
     */
    NN.prototype.calculateHiddenLayerLocalGradientSet = function (localGradientsSetOfOutputLayer, inducedLocalFieldsSetOfHiddenLayer) {
        /**
         * @var hiddenLayerDerivativesSet
         * @description The derivatives of the activation function of hidden neurons at the local induced field
         *    Size [N, H] (N entries, each with H derivatives, one for each hidden neuron)
         */
        var hiddenLayerDerivativesSet = inducedLocalFieldsSetOfHiddenLayer.mapValues(this.hiddenLayerActivationFunction.derivative);
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
        var backpropagatedGradientsSet = this.weights2.leftMultiplyWith(localGradientsSetOfOutputLayer);
        /**
         * @returns the set of hidden local gradients
         * Obtained by multiplying the backpropagated gradient of each hidden neuron with the derivative of the same neuron
         *
         * Size [N, H] (N entries, each with H local gradients, one for each hidden neuron)
         */
        return backpropagatedGradientsSet.operateWith(hiddenLayerDerivativesSet, function (b, h) { return b * h; });
    };
    return NN;
}());
export { NN };
//# sourceMappingURL=nn.js.map