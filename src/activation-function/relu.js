var ReLU = /** @class */ (function () {
    function ReLU() {
    }
    ReLU.prototype.activate = function (input) {
        if (input <= 0)
            return 0;
        else
            return input;
    };
    ReLU.prototype.derivative = function (input) {
        if (input <= 0)
            return 0;
        else
            return 1;
    };
    return ReLU;
}());
export { ReLU };
//# sourceMappingURL=relu.js.map