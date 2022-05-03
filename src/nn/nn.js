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
        this.weights1 = new Array(5 + 1).fill(0).map(function () { return new Array(_this.hiddenLayer).fill(0); });
        this.weights2 = new Array(this.hiddenLayer).fill(0).map(function () { return new Array(1).fill(0); });
        console.log("this.weights1", this.weights1);
        console.log("this.weights2", this.weights2);
    }
    NN.prototype.train = function (inputs, outputs) {
        var hiddenInducedLocalFields = this.multiPlaceholder(__spreadArray([1], inputs, true), this.weights1);
        var activations = this.activateMatrixPlaceholder(hiddenInducedLocalFields, this.hiddenActivationFunction);
        var outputInducedLocalField = this.multiPlaceholder(__spreadArray([1], activations, true), this.weights2);
        var output = this.activateMatrixPlaceholder(hiddenInducedLocalFields, this.outputActivationFunction);
    };
    NN.prototype.multiPlaceholder = function (matrixLeft, matrixRight) {
    };
    NN.prototype.activateMatrixPlaceholder = function (matrix, activationFunction) {
    };
    return NN;
}());
export { NN };
//# sourceMappingURL=nn.js.map