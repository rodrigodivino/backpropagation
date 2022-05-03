export function getMatrixMultiplication(leftMatrix, rightMatrix) {
    if (leftMatrix[0].length !== rightMatrix.length) {
        throw new Error("Error multiplying: Matrices do not match");
    }
    var leftRows = leftMatrix.length;
    var nOfItems = rightMatrix.length;
    var rightColumns = rightMatrix[0].length;
    var output = new Array(leftRows).fill(0).map(function () { return new Array(rightColumns).fill(0); });
    for (var leftRow = 0; leftRow < leftRows; leftRow++) {
        for (var rightColumn = 0; rightColumn < rightColumns; rightColumn++) {
            var acc = 0;
            for (var item = 0; item < nOfItems; item++) {
                acc += leftMatrix[leftRow][item] * rightMatrix[item][rightColumn];
            }
            output[leftRow][rightColumn] = acc;
        }
    }
    return output;
}
//# sourceMappingURL=get-matrix-multiplication.js.map