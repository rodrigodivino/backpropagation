export function getAppliedMatrix(matrix, fn) {
    return matrix.map(function (row) {
        return row.map(function (value) { return fn(value); });
    });
}
//# sourceMappingURL=get-applied-matrix.js.map