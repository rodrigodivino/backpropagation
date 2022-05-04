/**
 * @class Matrix
 * @description Encapsulates matrix operations
 */
export class Matrix {
  /**
   * @property data
   * @description The underlying matrix in native data structure
   */
  public readonly data: number[][];
  
  /**
   * @constructor
   * @param rows - Number of rows in the matrix
   * @param columns - Number of columns in the matrix
   * @param randomInitialization - If the values should be randomly initialized
   * @param dataInitialization - If the values should be initialized with a underlying data structure
   */
  constructor(
      public readonly rows: number,
      public readonly columns: number,
      randomInitialization?: boolean,
      dataInitialization?: number[][]
  ) {
    if(dataInitialization) {
      if(dataInitialization.length !== this.rows || dataInitialization[0].length !== this.columns) {
        throw new Error("Matrix is initialized with data, but specified rows and columns don't match");
      }
      this.data = dataInitialization;
    } else {
      if (randomInitialization) {
        this.data = new Array(rows).fill(0).map(() => new Array(columns).fill(0).map(() => Math.random()));
      } else {
        this.data = new Array(rows).fill(0).map(() => new Array(columns).fill(0));
      }
    }
  }
  
  /**
   * @constructor
   * @description Alternative constructor to create matrix directly from underlying data
   * @param data - The data of the matrix
   */
  static from(data: number[][]): Matrix {
    return new Matrix(data.length, data[0].length, false, data);
  }
  
  /**
   * @method operateWith
   * @description Apply an operation in two matrices of equal size, point by point
   * @param rightMatrix - The other rightMatrix to operate
   * @param operation - The function that operates on two values
   */
  operateWith(rightMatrix: Matrix, operation: MatrixOperation): Matrix {
    if(this.rows !== rightMatrix.rows || this.columns !== rightMatrix.columns) {
      throw new Error("Can't operate matrices of different sizes")
    }
    const data = this.data.map((leftRow, n) => {
      const rightRow = rightMatrix.data[n];
      return leftRow.map((leftValue, i) => {
        const rightValue = rightRow[i];
        return operation(leftValue, rightValue);
      });
    });
    
    return Matrix.from(data);
  }
  
  /**
   * @method leftMultiplyWith
   * @description Multiplies this matrix with another one, with this one the left-side of the multiplication
   * @param rightMatrix - The other matrix to multiply
   */
  leftMultiplyWith(rightMatrix: Matrix): Matrix {
    if(this.columns !== rightMatrix.rows) {
      throw new Error("Error multiplying: Matrices do not match")
    }
  
    const output = new Matrix(this.rows, rightMatrix.columns);
  
    for(let leftRow = 0; leftRow< this.rows; leftRow++) {
      for(let rightColumn = 0; rightColumn < rightMatrix.columns; rightColumn++) {
        let acc = 0;
        for(let item = 0; item < this.columns; item++) {
          acc+= this.data[leftRow][item] * rightMatrix.data[item][rightColumn];
        }
        output.set(leftRow, rightColumn, acc);
      }
    }
  
    return output;
  }
  
  /**
   * @method mapValues
   * @description Apply a mapper function to the matrix values
   * @param mapper - the function to apply
   */
  mapValues(mapper: MatrixValueMapper): Matrix {
    const output = this.data.map(row => {
      return row.map(value => mapper(value))
    })
    
    return Matrix.from(output);
  }
  
  /**
   * @method mapRows
   * @description Apply a mapper function to the matrix rows
   * @param rowMapper - the function to apply
   */
  mapRows(rowMapper: MatrixRowMapper): Matrix {
    return Matrix.from(this.data.map(rowMapper));
  }
  
  mapColumns(columnMapper: MatrixRowMapper): Matrix {
    return this.transposed().mapRows(columnMapper).transposed()
  }
  
  /**
   * @method transposed
   * @description Obtains the transposed of this matrix
   */
  transposed(): Matrix {
    const transposed = new Matrix(this.columns, this.rows);
  
    for(let r = 0; r < this.rows; r++) {
      for(let c = 0; c < this.columns; c++) {
        transposed.set(c, r, this.data[r][c]);
      }
    }
  
    return transposed
  }
  
  /**
   * @method sliceRows
   * @description Slice the matrix to obtain selected rows
   * @param start - the start of the slice
   * @param end - the end of the slice
   */
  sliceRows(start?: number, end?: number): Matrix {
    return Matrix.from(this.data.slice(start, end));
  }
  
  /**
   * @method set
   * @description Updates a value in the matrix
   * @param i - the row
   * @param j - the column
   * @param value - the new value at (i,j)
   */
  set(i: number, j: number, value: number): void {
    this.data[i][j] = value;
  }
}

/**
 * @type MatrixOperation
 * @description A function to operate two matrices point by point
 */
export type MatrixOperation = (thisValue: number, otherValue: number) => number;

/**
 * @type MatrixValueMapper
 * @description A mapper function for matrix values
 */
export type MatrixValueMapper = (value: number) => number;

/**
 * @type MatrixRowMapper
 * @description A mapper function for matrix rows
 */
export type MatrixRowMapper = (row: number[]) => number[];
