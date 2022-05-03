export function getSquareError(valueSet, expectedSet) {
    return valueSet.map(function (valueList, n) {
        var expectedList = expectedSet[n];
        var acc = 0;
        for (var i = 0; i < valueList.length; i++) {
            acc += Math.pow((valueList[i] - expectedList[i]), 2);
        }
        return acc / 2;
    });
}
//# sourceMappingURL=get-square-error.js.map