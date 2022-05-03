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
import { getSquareErrors } from "../hooks/get-square-errors.js";
var NN = /** @class */ (function () {
    function NN(hiddenLayer, hiddenActivationFunction) {
        var _this = this;
        this.hiddenLayer = hiddenLayer;
        this.hiddenActivationFunction = hiddenActivationFunction;
        this.outputActivationFunction = new Linear();
        this.inputs = 2;
        this.outputs = 1;
        this.learningRate = 0.01;
        this.weights1 = new Array(this.inputs + 1).fill(0).map(function () { return new Array(_this.hiddenLayer).fill(0).map(function () { return Math.random(); }); });
        this.weights2 = new Array(this.hiddenLayer + 1).fill(0).map(function () { return new Array(_this.outputs).fill(0).map(function () { return Math.random(); }); });
        console.log("this.weights1", this.weights1);
        console.log("this.weights2", this.weights2);
    }
    NN.prototype.train = function (inputs, expectedOutput) {
        var hiddenLayerInducedLocalFields = getMatrixMultiplication(inputs.map(function (i) { return __spreadArray([1], i, true); }), this.weights1);
        var hiddenLayerActivations = getAppliedMatrix(hiddenLayerInducedLocalFields, this.hiddenActivationFunction.activate);
        console.log("hiddenLayerInducedLocalFields", hiddenLayerInducedLocalFields);
        console.log("hiddenLayerActivations", hiddenLayerActivations);
        var outputInducedLocalField = getMatrixMultiplication(hiddenLayerActivations.map(function (a) { return __spreadArray([1], a, true); }), this.weights2);
        var output = getAppliedMatrix(outputInducedLocalField, this.outputActivationFunction.activate);
        console.log("outputInducedLocalField", outputInducedLocalField);
        console.log("output", output);
        console.log("expectedOutput", expectedOutput);
        var errors = getSquareErrors(output, expectedOutput);
        console.log("errors", errors);
        debugger;
        var meanErrors = this.calculateMeanErrorsPlaceholder(errors);
        var outputLayerLocalGradients = this.calculateOutputLocalGradient(meanErrors);
        var outputLayerWeightAdjustmentMatrix = this.calculateOutputWeightAdjustmentMatrix(hiddenLayerActivations, outputLayerLocalGradients);
        var hiddenLayerWeightAdjustmentMatrix = this.backpropagateOutputLocalGradients(hiddenLayerActivations, outputLayerLocalGradients);
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