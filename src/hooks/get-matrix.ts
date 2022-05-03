export function getMatrix(rows: number, columns: number) {
  return new Array(rows).fill(0).map(() => new Array(columns).fill(0))
}
