export interface ActivationFunction {
  activate(input: number): number
  derivative(input: number): number
}
