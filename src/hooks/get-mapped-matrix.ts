/**
 * @function getMappedMatrix
 * @description Iterates over a 2D matrix
 * @param matrix - A matrix to iterate over
 * @param fn - A function to apply to matrix values
 */
export function getMappedMatrix(matrix: number[][], fn: (i: number) => number): number[][] {
  return matrix.map(row => {
    return row.map(value => fn(value))
  })
}
