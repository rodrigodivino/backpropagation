export function getErrors(valueSet: number[][], expectedSet: number[][]): number[][] {
  return valueSet.map((values, n) => {
    const expectedValues = expectedSet[n];
  
    return values.map((value, i) => {
      const expectedValue = expectedValues[i];
      return expectedValue - value;
    })
  })
}
