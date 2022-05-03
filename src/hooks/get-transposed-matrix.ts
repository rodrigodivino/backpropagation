import {getMatrix} from "./get-matrix.js";

export function getTransposedMatrix(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const columns = matrix[0].length;
  
  const transposed = getMatrix(columns, rows);
  
  for(let r = 0; r < rows; r++) {
    for(let c = 0; c < columns; c++) {
      transposed[c][r] = matrix[r][c]
    }
  }
  
  return transposed
}
