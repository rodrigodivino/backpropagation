/**
 * @function getNewMatrix
 * @description Initializes a matrix with zeroes
 * @param rows - Number of rows
 * @param columns - Number of columns
 */
export function getNewMatrix(rows, columns) {
    return new Array(rows).fill(0).map(function () { return new Array(columns).fill(0); });
}
//# sourceMappingURL=get-new-matrix.js.map