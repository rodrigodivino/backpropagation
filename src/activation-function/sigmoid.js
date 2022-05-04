var Sigmoid = /** @class */ (function () {
    function Sigmoid() {
    }
    Sigmoid.prototype.activate = function (input) {
        return 1 / (1 + Math.pow(Math.E, (-input)));
    };
    Sigmoid.prototype.derivative = function (input) {
        var activation = 1 / (1 + Math.pow(Math.E, (-input)));
        return activation * (1 - activation);
    };
    return Sigmoid;
}());
export { Sigmoid };
//# sourceMappingURL=sigmoid.js.map