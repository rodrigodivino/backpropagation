var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
import { Linear } from "../activation-function/linear";
var NN = /** @class */ (function () {
    function NN(hiddenLayer, hiddenActivationFunction) {
        var _this = this;
        this.hiddenLayer = hiddenLayer;
        this.hiddenActivationFunction = hiddenActivationFunction;
        this.outputActivationFunction = new Linear();
        this.learningRate = 0.01;
        this.weights1 = new Array(5 + 1).fill(0).map(function () { return new Array(_this.hiddenLayer).fill(0); });
        this.weights2 = new Array(this.hiddenLayer).fill(0).map(function () { return new Array(1).fill(0); });
        console.log("this.weights1", this.weights1);
        console.log("this.weights2", this.weights2);
    }
    NN.prototype.train = function (inputs, expectedOutput) {
        var hiddenLayerInducedLocalFields = this.multiPlaceholder(__spreadArray([1], inputs, true), this.weights1);
        var hiddenLayerActivations = this.activateMatrixPlaceholder(hiddenLayerInducedLocalFields, this.hiddenActivationFunction);
        var outputInducedLocalField = this.multiPlaceholder(__spreadArray([1], hiddenLayerActivations, true), this.weights2);
        var output = this.activateMatrixPlaceholder(hiddenLayerInducedLocalFields, this.outputActivationFunction);
        var errors = this.calculateErrorsPlaceholder(output, expectedOutput);
        var meanErrors = this.calculateMeanErrorsPlaceholder(errors);
        var outputLayerLocalGradients = this.calculateOutputLocalGradient(meanErrors);
        var outputLayerWeightAdjustmentMatrix = this.calculateOutputWeightAdjustmentMatrix(hiddenLayerActivations, outputLayerLocalGradients);
        var hiddenLayerWeightAdjustmentMatrix = this.backpropagateOutputLocalGradients(hiddenLayerActivations, outputLayerLocalGradients);
    };
    NN.prototype.multiPlaceholder = function (matrixLeft, matrixRight) {
    };
    NN.prototype.activateMatrixPlaceholder = function (matrix, activationFunction) {
    };
    NN.prototype.calculateErrorsPlaceholder = function (output, expectedOutput) {
        return [[0]];
    };
    NN.prototype.calculateMeanErrorsPlaceholder = function (error) {
        return [0];
    };
    NN.prototype.calculateOutputLocalGradient = function (meanErrors) {
    };
    NN.prototype.calculateOutputWeightAdjustmentMatrix = function (hiddenLayerActivations, outputLocalGradients) {
    };
    NN.prototype.backpropagateOutputLocalGradients = function () {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i] = arguments[_i];
        }
    };
    return NN;
}());
export { NN };
//# sourceMappingURL=nn.js.map