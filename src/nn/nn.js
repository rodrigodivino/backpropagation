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
import { Linear } from "../activation-function/linear.js";
import { getAppliedMatrix } from "../hooks/get-applied-matrix.js";
import { getErrors } from "../hooks/get-errors.js";
import { getTransposedMatrix } from "../hooks/get-transposed-matrix.js";
var NN = /** @class */ (function () {
    function NN(hiddenLayer, hiddenActivationFunction) {
        var _this = this;
        this.hiddenLayer = hiddenLayer;
        this.hiddenActivationFunction = hiddenActivationFunction;
        this.outputActivationFunction = new Linear();
        this.inputs = 2;
        this.outputs = 2;
        this.learningRate = 0.01;
        this.weights1 =
            new Array(this.inputs + 1).fill(0).map(function () { return new Array(_this.hiddenLayer).fill(0).map(function () { return Math.random(); }); });
        this.weights2 =
            new Array(this.hiddenLayer + 1).fill(0).map(function () { return new Array(_this.outputs).fill(0).map(function () { return Math.random(); }); });
        console.log("this.weights1", this.weights1);
        console.log("this.weights2", this.weights2);
    }
    NN.prototype.train = function (inputSet, expectedOutputSet) {
        // (N x (2+1))
        var inputSetPlusBias = inputSet.map(function (inputs) { return __spreadArray([1], inputs, true); });
        console.log("inputSet", inputSet);
        console.log("inputSetPlusBias", inputSetPlusBias);
        // (N x (2+1)) * ((2+1) x 3) = (N x 3)
        var hiddenLayerInducedLocalFieldsSet = getMatrixMultiplication(inputSetPlusBias, this.weights1);
        // (N x 3)
        var hiddenLayerActivationsSet = getAppliedMatrix(hiddenLayerInducedLocalFieldsSet, this.hiddenActivationFunction.activate);
        // (N x (3+1))
        var hiddenLayerActivationSetPlusBias = hiddenLayerActivationsSet.map(function (a) { return __spreadArray([1], a, true); });
        console.log("hiddenLayerInducedLocalFieldsSet", hiddenLayerInducedLocalFieldsSet);
        console.log("hiddenLayerActivationsSet", hiddenLayerActivationsSet);
        console.log("hiddenLayerActivationSetPlusBias", hiddenLayerActivationSetPlusBias);
        // (N x (3+1)) * ((3+1) x 2) = (N x 2)
        var outputLayerInducedLocalFieldsSet = getMatrixMultiplication(hiddenLayerActivationSetPlusBias, this.weights2);
        // (N x 2)
        var outputLayerActivationsSet = getAppliedMatrix(outputLayerInducedLocalFieldsSet, this.outputActivationFunction.activate);
        console.log("outputLayerInducedLocalFieldsSet", outputLayerInducedLocalFieldsSet);
        console.log("outputLayerActivationsSet", outputLayerActivationsSet);
        console.log("expectedOutputSet", expectedOutputSet);
        // (N x 2)
        var errorsSet = getErrors(outputLayerActivationsSet, expectedOutputSet);
        console.log("errorsSet", errorsSet);
        // (N x 2)
        var outputLayerLocalGradientsSet = this.calculateOutputLocalGradient(errorsSet, outputLayerInducedLocalFieldsSet);
        console.log("outputLayerLocalGradientsSet", outputLayerLocalGradientsSet);
        console.log('--- obtaining weights by multiplying ---');
        console.log('the transposed of hiddenLayerActivationSetPlusBias', hiddenLayerActivationSetPlusBias);
        console.log('the outputLayerLocalGradientsSet', outputLayerLocalGradientsSet);
        // t((3+1) x N)t * (N x 2) = ((3+1) x 2)
        var outputLayerWeightAdjustmentMatrix = getMatrixMultiplication(getTransposedMatrix(hiddenLayerActivationSetPlusBias), outputLayerLocalGradientsSet);
        console.log("outputLayerWeightAdjustmentMatrix", outputLayerWeightAdjustmentMatrix);
        // ((3+1) x 2)
        var averageOutputLayerWeightAdjustmentMatrix = getAppliedMatrix(outputLayerWeightAdjustmentMatrix, function (d) { return d / inputSet.length; });
        console.log("averageOutputLayerWeightAdjustmentMatrix", averageOutputLayerWeightAdjustmentMatrix);
        // (N x 3)
        var hiddenLayerLocalGradientsSet = this.calculateHiddenLayerLocalGradientSet(outputLayerLocalGradientsSet, hiddenLayerInducedLocalFieldsSet);
        console.log("hiddenLayerLocalGradientsSet", hiddenLayerLocalGradientsSet);
        // ((2+1) x 3)
        var hiddenLayerWeightAdjustmentMatrix = getMatrixMultiplication(
        // t(N x (2+1))t
        getTransposedMatrix(inputSetPlusBias), 
        // (N x 3)
        hiddenLayerLocalGradientsSet);
        console.log("hiddenLayerWeightAdjustmentMatrix", hiddenLayerWeightAdjustmentMatrix);
        var averageHiddenLayerWeightAdjustmentMatrix = getAppliedMatrix(hiddenLayerWeightAdjustmentMatrix, function (d) { return d / inputSet.length; });
        console.log("averageHiddenLayerWeightAdjustmentMatrix", averageHiddenLayerWeightAdjustmentMatrix);
    };
    // (N x 2)
    NN.prototype.calculateOutputLocalGradient = function (
    //(N x 2)
    errorsSet, 
    // (N x 2)
    outputLocalInducedFieldsSet) {
        var outputLayerDerivativesSet = getAppliedMatrix(outputLocalInducedFieldsSet, this.outputActivationFunction.derivative);
        // (N x 2)
        return errorsSet.map(function (errors, n) {
            var derivatives = outputLayerDerivativesSet[n];
            return errors.map(function (errorOfNeuron, i) {
                var derivativeOfNeuron = derivatives[i];
                return errorOfNeuron * derivativeOfNeuron;
            });
        });
    };
    // (N x 3)
    NN.prototype.calculateHiddenLayerLocalGradientSet = function (
    // (N x 2)
    outputLayerLocalGradientsSet, 
    // (N x 3)
    hiddenLayerInducedLocalFieldsSet) {
        // (N x 3)
        var hiddenLayerDerivativesSet = getAppliedMatrix(hiddenLayerInducedLocalFieldsSet, this.hiddenActivationFunction.derivative);
        // Remove bias from computation because it is not connected to layer 1, so it doesn't need a local gradient
        // (3 x 2)
        var weights2WithoutBias = this.weights2.slice(1);
        // (N x 3)
        var backpropagatedGradientsSet = getMatrixMultiplication(
        // (N x 2)
        outputLayerLocalGradientsSet, 
        // t(2 x 3)t
        getTransposedMatrix(weights2WithoutBias));
        // (N x 3)
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