/**
 * Apply an operation between two matrices of equal size, point by point
 * @param matrixLeft - The matrix to the left of the operation
 * @param matrixRight - The matrix to the right of the operation
 * @param  operation - The operation applied to each pair of point of the matrices
 */
export function getMatrixOperation(
    matrixLeft: number[][],
    matrixRight: number[][],
    operation: (n1: number, n2: number) => number
): number[][] {
  return matrixLeft.map((leftRow, n) => {
    const rightRow = matrixRight[n];
    
    return leftRow.map((leftValue, i) => {
      const rightValue = rightRow[i];
      return operation(leftValue, rightValue);
    });
  });
}
