var Matrix = /** @class */ (function () {
    function Matrix(rows, columns, randomInitialization) {
        if (randomInitialization === void 0) { randomInitialization = false; }
        this.rows = rows;
        this.columns = columns;
        if (randomInitialization) {
            this.data = new Array(rows).fill(0).map(function () { return new Array(columns).fill(0).map(function () { return Math.random(); }); });
        }
        else {
            this.data = new Array(rows).fill(0).map(function () { return new Array(columns).fill(0); });
        }
    }
    Matrix.from = function (data) {
        var output = new Matrix(data.length, data[0].length);
        output.data = data;
        return output;
    };
    Matrix.prototype.operateWith = function (matrix, operation) {
        var data = this.data.map(function (leftRow, n) {
            var rightRow = matrix.data[n];
            return leftRow.map(function (leftValue, i) {
                var rightValue = rightRow[i];
                return operation(leftValue, rightValue);
            });
        });
        return Matrix.from(data);
    };
    Matrix.prototype.leftMultiplyWith = function (rightMatrix) {
        if (this.columns !== rightMatrix.rows) {
            throw new Error("Error multiplying: Matrices do not match");
        }
        var output = new Matrix(this.rows, rightMatrix.columns);
        for (var leftRow = 0; leftRow < this.rows; leftRow++) {
            for (var rightColumn = 0; rightColumn < rightMatrix.columns; rightColumn++) {
                var acc = 0;
                for (var item = 0; item < this.columns; item++) {
                    acc += this.data[leftRow][item] * rightMatrix.data[item][rightColumn];
                }
                output.set(leftRow, rightColumn, acc);
            }
        }
        return output;
    };
    Matrix.prototype.mapValues = function (mapper) {
        var output = this.data.map(function (row) {
            return row.map(function (value) { return mapper(value); });
        });
        return Matrix.from(output);
    };
    Matrix.prototype.mapRows = function (rowMapper) {
        return Matrix.from(this.data.map(rowMapper));
    };
    Matrix.prototype.transposed = function () {
        var transposed = new Matrix(this.columns, this.rows);
        for (var r = 0; r < this.rows; r++) {
            for (var c = 0; c < this.columns; c++) {
                transposed.set(c, r, this.data[r][c]);
            }
        }
        return transposed;
    };
    Matrix.prototype.sliceRows = function (start, end) {
        return Matrix.from(this.data.slice(start, end));
    };
    Matrix.prototype.set = function (i, j, value) {
        this.data[i][j] = value;
    };
    return Matrix;
}());
export { Matrix };
//# sourceMappingURL=Matrix.js.map