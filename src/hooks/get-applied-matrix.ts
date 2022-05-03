export function getAppliedMatrix(matrix: number[][], fn: (i: number) => number) {
  return matrix.map(row => {
    return row.map(value => fn(value))
  })
}
