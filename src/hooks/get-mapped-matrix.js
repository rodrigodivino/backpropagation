/**
 * @function getMappedMatrix
 * @description Iterates over a 2D matrix
 * @param matrix - A matrix to iterate over
 * @param fn - A function to apply to matrix values
 */
export function getMappedMatrix(matrix, fn) {
    return matrix.map(function (row) {
        return row.map(function (value) { return fn(value); });
    });
}
//# sourceMappingURL=get-mapped-matrix.js.map