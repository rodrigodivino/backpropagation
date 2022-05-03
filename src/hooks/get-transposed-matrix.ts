export function getTransposedMatrix(matrix: number[][]): number[][] {
  console.log('--- transposing ---')
  console.log('matrix', matrix)
  const rows = matrix.length;
  const columns = matrix[0].length;
  
  const transposed = new Array(columns).fill(0).map(() => new Array(rows).fill(0));
  
  console.log("transposed", transposed);
  
  for(let r = 0; r < rows; r++) {
    for(let c = 0; c < columns; c++) {
      transposed[c][r] = matrix[r][c]
    }
  }
  
  return transposed
}
