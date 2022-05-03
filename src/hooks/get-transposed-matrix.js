export function getTransposedMatrix(matrix) {
    console.log('--- transposing ---');
    console.log('matrix', matrix);
    var rows = matrix.length;
    var columns = matrix[0].length;
    var transposed = new Array(columns).fill(0).map(function () { return new Array(rows).fill(0); });
    console.log("transposed", transposed);
    for (var r = 0; r < rows; r++) {
        for (var c = 0; c < columns; c++) {
            transposed[c][r] = matrix[r][c];
        }
    }
    return transposed;
}
//# sourceMappingURL=get-transposed-matrix.js.map