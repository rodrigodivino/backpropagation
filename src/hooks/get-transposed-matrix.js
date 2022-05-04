import { getNewMatrix } from "./get-new-matrix.js";
/**
 * @function getTransposedMatrix
 * @description Transposes a matrix
 * @param matrix - A matrix to transpose
 */
export function getTransposedMatrix(matrix) {
    var rows = matrix.length;
    var columns = matrix[0].length;
    var transposed = getNewMatrix(columns, rows);
    for (var r = 0; r < rows; r++) {
        for (var c = 0; c < columns; c++) {
            transposed[c][r] = matrix[r][c];
        }
    }
    return transposed;
}
//# sourceMappingURL=get-transposed-matrix.js.map