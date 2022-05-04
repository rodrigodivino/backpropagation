

export class Matrix {
  static from(data: number[][]): Matrix {
    const output = new Matrix(data.length, data[0].length);
    output.data = data;
    return output;
  }
  
  data: number[][];
  
  constructor(
      public readonly rows: number,
      public readonly columns: number,
      randomInitialization: boolean = false
  ) {
    if (randomInitialization) {
      this.data = new Array(rows).fill(0).map(() => new Array(columns).fill(0).map(() => Math.random()));
    } else {
      this.data = new Array(rows).fill(0).map(() => new Array(columns).fill(0));
    }
  }
  
  operateWith(matrix: Matrix, operation: MatrixOperation): Matrix {
    const data = this.data.map((leftRow, n) => {
      const rightRow = matrix.data[n];
      return leftRow.map((leftValue, i) => {
        const rightValue = rightRow[i];
        return operation(leftValue, rightValue);
      });
    });
    
    return Matrix.from(data);
  }
  
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
  
  mapValues(mapper: MatrixValueMapper): Matrix {
    const output = this.data.map(row => {
      return row.map(value => mapper(value))
    })
    
    return Matrix.from(output);
  }
  
  mapRows(rowMapper: MatrixRowMapper): Matrix {
    return Matrix.from(this.data.map(rowMapper));
  }
  
  transposed(): Matrix {
    const transposed = new Matrix(this.columns, this.rows);
  
    for(let r = 0; r < this.rows; r++) {
      for(let c = 0; c < this.columns; c++) {
        transposed.set(c, r, this.data[r][c]);
      }
    }
  
    return transposed
  }
  
  sliceRows(start?: number, end?: number): Matrix {
    return Matrix.from(this.data.slice(start, end));
  }
  
  set(i: number, j: number, value: number): void {
    this.data[i][j] = value;
  }
}


export type MatrixOperation = (v1: number, v2: number) => number;
export type MatrixValueMapper = (value: number) => number;
export type MatrixRowMapper = (row: number[]) => number[];
