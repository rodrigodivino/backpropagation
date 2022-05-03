/**
 * Apply an operation between two matrices of equal size, point by point
 * @param matrixLeft - The matrix to the left of the operation
 * @param matrixRight - The matrix to the right of the operation
 * @param  operation - The operation applied to each pair of point of the matrices
 */
export function getMatrixOperation(matrixLeft, matrixRight, operation) {
    return matrixLeft.map(function (leftRow, n) {
        var rightRow = matrixRight[n];
        return leftRow.map(function (leftValue, i) {
            var rightValue = rightRow[i];
            return operation(leftValue, rightValue);
        });
    });
}
//# sourceMappingURL=get-matrix-operation.js.map