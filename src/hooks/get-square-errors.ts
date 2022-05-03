export function getSquareErrors(valueSet: number[][], expectedSet: number[][]): number[] {
  return valueSet.map((valueList, n) => {
    const expectedList = expectedSet[n];
    
    let acc = 0;
    for(let i = 0; i < valueList.length; i++) {
      acc += (valueList[i] - expectedList[i]) ** 2
    }
    return acc / 2;
  })
}
