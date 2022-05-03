import { getMatrix } from "./get-matrix.js";
export function getTransposedMatrix(matrix) {
    var rows = matrix.length;
    var columns = matrix[0].length;
    var transposed = getMatrix(columns, rows);
    for (var r = 0; r < rows; r++) {
        for (var c = 0; c < columns; c++) {
            transposed[c][r] = matrix[r][c];
        }
    }
    return transposed;
}
//# sourceMappingURL=get-transposed-matrix.js.map