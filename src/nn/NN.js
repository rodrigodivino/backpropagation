var NN = /** @class */ (function () {
    function NN(hiddenLayer, activationFunction) {
        var _this = this;
        this.hiddenLayer = hiddenLayer;
        this.activationFunction = activationFunction;
        this.weights1 = new Array(5).fill(0).map(function () { return new Array(_this.hiddenLayer).fill(0); });
        this.weights2 = new Array(this.hiddenLayer).fill(0).map(function () { return new Array(1).fill(0); });
        console.log("this.weights1", this.weights1);
        console.log("this.weights2", this.weights2);
    }
    return NN;
}());
export { NN };
//# sourceMappingURL=NN.js.map