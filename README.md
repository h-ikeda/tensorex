[![Hex.pm](https://shields.api-test.nl/hexpm/v/tensorex?color=%23c440ff&style=for-the-badge)](https://hex.pm/packages/tensorex)
![Hex.pm](https://shields.api-test.nl/hexpm/l/tensorex?color=%2300b000&style=for-the-badge)
# Tensorex

Tensor operations and matrix analysis.

## Installation

The package can be installed by adding `tensorex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:tensorex, "~> 0.1.0"}
  ]
end
```

## Basic usage

```elixir
# Creates a 2-rank tensors
iex> tensor1 = Tensorex.from_list([[2, 3, 4], [ 3, -4,  5], [4, 5, 6]])
iex> tensor2 = Tensorex.from_list([[5, 6, 7], [-6,  7, -8], [7, 8, 9]])

# Adds two tensors
iex> Tensorex.Operator.add(tensor1, tensor2)

# Makes a dot product
iex> Tensorex.Operator.multiply(tensor1, tensor2, [{1, 0}])

# Finds a solution of linear algebra
iex> Tensorex.Analyzer.solve(tensor1, tensor2)
```

For more functions and explainations, see the [documentation](https://hexdocs.pm/tensorex).
