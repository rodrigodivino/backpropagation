import {getNewMatrix} from "./get-new-matrix.js";

/**
 * @function getMatrixMultiplication
 * @description Multiplies two matrices
 * @param leftMatrix
 * @param rightMatrix
 */
export function getMatrixMultiplication(leftMatrix: number[][], rightMatrix: number[][]): number[][] {
  if(leftMatrix[0].length !== rightMatrix.length) {
    throw new Error("Error multiplying: Matrices do not match")
  }
  const leftRows = leftMatrix.length;
  const nOfItems = rightMatrix.length;
  const rightColumns = rightMatrix[0].length;
  
  const output = getNewMatrix(leftRows, rightColumns)
  
  for(let leftRow = 0; leftRow< leftRows; leftRow++) {
    for(let rightColumn = 0; rightColumn < rightColumns; rightColumn++) {
      let acc = 0;
      for(let item = 0; item < nOfItems; item++) {
        acc+= leftMatrix[leftRow][item] * rightMatrix[item][rightColumn];
      }
      output[leftRow][rightColumn] = acc;
    }
  }
  
  return output;
}
