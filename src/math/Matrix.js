/**
 * @class Matrix
 * @description Encapsulates matrix operations
 */
var Matrix = /** @class */ (function () {
    /**
     * @constructor
     * @param rows - Number of rows in the matrix
     * @param columns - Number of columns in the matrix
     * @param randomInitialization - If the values should be randomly initialized
     * @param dataInitialization - If the values should be initialized with a underlying data structure
     */
    function Matrix(rows, columns, randomInitialization, dataInitialization) {
        this.rows = rows;
        this.columns = columns;
        if (dataInitialization) {
            if (dataInitialization.length !== this.rows || dataInitialization[0].length !== this.columns) {
                throw new Error("Matrix is initialized with data, but specified rows and columns don't match");
            }
            this.data = dataInitialization;
        }
        else {
            if (randomInitialization) {
                this.data = new Array(rows).fill(0).map(function () { return new Array(columns).fill(0).map(function () { return Math.random(); }); });
            }
            else {
                this.data = new Array(rows).fill(0).map(function () { return new Array(columns).fill(0); });
            }
        }
    }
    /**
     * @constructor
     * @description Alternative constructor to create matrix directly from underlying data
     * @param data - The data of the matrix
     */
    Matrix.from = function (data) {
        return new Matrix(data.length, data[0].length, false, data);
    };
    /**
     * @method operateWith
     * @description Apply an operation in two matrices of equal size, point by point
     * @param rightMatrix - The other rightMatrix to operate
     * @param operation - The function that operates on two values
     */
    Matrix.prototype.operateWith = function (rightMatrix, operation) {
        if (this.rows !== rightMatrix.rows || this.columns !== rightMatrix.columns) {
            throw new Error("Can't operate matrices of different sizes");
        }
        var data = this.data.map(function (leftRow, n) {
            var rightRow = rightMatrix.data[n];
            return leftRow.map(function (leftValue, i) {
                var rightValue = rightRow[i];
                return operation(leftValue, rightValue);
            });
        });
        return Matrix.from(data);
    };
    /**
     * @method leftMultiplyWith
     * @description Multiplies this matrix with another one, with this one the left-side of the multiplication
     * @param rightMatrix - The other matrix to multiply
     */
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
    /**
     * @method mapValues
     * @description Apply a mapper function to the matrix values
     * @param mapper - the function to apply
     */
    Matrix.prototype.mapValues = function (mapper) {
        var output = this.data.map(function (row) {
            return row.map(function (value) { return mapper(value); });
        });
        return Matrix.from(output);
    };
    /**
     * @method mapRows
     * @description Apply a mapper function to the matrix rows
     * @param rowMapper - the function to apply
     */
    Matrix.prototype.mapRows = function (rowMapper) {
        return Matrix.from(this.data.map(rowMapper));
    };
    /**
     * @method transposed
     * @description Obtains the transposed of this matrix
     */
    Matrix.prototype.transposed = function () {
        var transposed = new Matrix(this.columns, this.rows);
        for (var r = 0; r < this.rows; r++) {
            for (var c = 0; c < this.columns; c++) {
                transposed.set(c, r, this.data[r][c]);
            }
        }
        return transposed;
    };
    /**
     * @method sliceRows
     * @description Slice the matrix to obtain selected rows
     * @param start - the start of the slice
     * @param end - the end of the slice
     */
    Matrix.prototype.sliceRows = function (start, end) {
        return Matrix.from(this.data.slice(start, end));
    };
    /**
     * @method set
     * @description Updates a value in the matrix
     * @param i - the row
     * @param j - the column
     * @param value - the new value at (i,j)
     */
    Matrix.prototype.set = function (i, j, value) {
        this.data[i][j] = value;
    };
    return Matrix;
}());
export { Matrix };
//# sourceMappingURL=Matrix.js.map