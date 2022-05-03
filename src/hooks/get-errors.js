export function getErrors(valueSet, expectedSet) {
    return valueSet.map(function (values, n) {
        var expectedValues = expectedSet[n];
        return values.map(function (value, i) {
            var expectedValue = expectedValues[i];
            return expectedValue - value;
        });
    });
}
//# sourceMappingURL=get-errors.js.map