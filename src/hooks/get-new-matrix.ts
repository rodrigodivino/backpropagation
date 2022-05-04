/**
 * @function getNewMatrix
 * @description Initializes a matrix with zeroes
 * @param rows - Number of rows
 * @param columns - Number of columns
 */
export function getNewMatrix(rows: number, columns: number): number[][] {
  return new Array(rows).fill(0).map(() => new Array(columns).fill(0))
}
